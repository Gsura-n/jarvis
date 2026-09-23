# Jarvis v2 Architecture

Status: living document, last updated September 15, 2026. Decisions are recorded in docs/adr/.
Target: production grade personal system, multi device from day one.

## 1. What we are building

Jarvis is a local AI orchestration gateway. It exposes one OpenAI compatible endpoint, decides how much model a request deserves, runs it on whichever of my machines is best placed to serve it, and returns an answer that has passed a set of quality gates. Everything runs on hardware I own. Cloud services are limited to free tier storage (Neo4j AuraDB, Neon Postgres) and can be swapped for local equivalents.

The v2 goal is not more features. It is that the existing pipeline becomes reliable, measurable, secure and reproducible, and that it can use both the Mac mini and the MacBook Pro without me thinking about it.

### Goals

The gateway routes each request to the smallest model that can do the job well, and escalates to a larger model on the MacBook Pro when the task calls for it. Both machines are used together when they are both online, and the mini alone keeps working when the laptop is closed or away. Every model output passes deterministic validation, and the expensive outputs pass a judge before they are accepted. Every request is traceable end to end. The repo installs from a clean clone with one command, has tests and CI, and has no secrets in it. A golden evaluation set gives a number for routing accuracy and answer quality that I can put in the README and re-run after every change.

### Non goals for v2

Multi user, accounts, hosted deployment, a custom UI (Open WebUI stays), cloud model fallbacks (designed for but not wired), and pipeline parallel splitting of one model across both machines. That last one is a separate experiment after v2 ships.

## 2. Hardware and roles

| Machine | Chip / RAM | Role | Runs |
|---|---|---|---|
| Mac mini M4, 16 GB | fast tokens/s on small models, always on | control plane and small tier | gateway, LiteLLM, Open WebUI, RAG watcher, Ollama with small models |
| MacBook Pro M1 Pro, 32 GB | more memory, sometimes away | large tier compute node | Ollama only, with large models, headless |

The mini is the brain and is always the entry point. The MacBook Pro is a worker that may or may not be reachable. Nothing on the mini depends on the laptop being present.

Model tiers, not devices, are the unit the gateway reasons about. A tier is "what class of model does this need" and a device is "who can serve that tier right now."

| Tier | Purpose | Mini deployment | MacBook Pro deployment |
|---|---|---|---|
| classify | intent, complexity, structured JSON | llama3.2:3b (resident) | not needed |
| embed | RAG embeddings | nomic-embed-text (resident) | not needed |
| general-small | chat, summaries | llama3.1:8b | same, as overflow |
| code-small | quick code questions | qwen2.5-coder:7b | same |
| reasoning-large | design, debugging logic | deepseek-r1:8b (degraded fallback) | qwen3:30b-a3b-instruct-2507-q4_K_M |
| code-large | implementation from a design | qwen2.5-coder:7b (fallback) | qwen3:30b-a3b-instruct-2507-q4_K_M (same resident model) |
| judge | scoring outputs | llama3.1:8b | same |

Decided September 15: the MacBook Pro serves exactly one large model, the Qwen3 30B-A3B MoE at Q4 (19 GB), for both reasoning-large and code-large. With a 26 GB GPU limit, any second model on that node (including qwen2.5-coder:14b at 9 GB) forces a reload of 10 to 20 seconds, which the decomposer's reasoning-then-code pattern would trigger constantly. qwen2.5-coder:14b is pulled on the laptop as a benchmark baseline and emergency fallback only, and is not in the regular routing mix. Small code work goes to the mini's qwen2.5-coder:7b. qwen3.6:35b-a3b was ruled out because its smallest quantization is 23 GB. qwen3-coder:30b is a candidate to benchmark later through the eval suite.

## 3. Architecture

```
Open WebUI / Continue / curl
            |
            v
   +-------------------+
   |  Jarvis Gateway   |  port 4001, API key required
   |  (FastAPI, mini)  |
   +-------------------+
     |      |       |
     |      |       +--> Device Registry (health, RAM, loaded models per node)
     |      +----------> Policy Engine (intent + complexity + availability -> deployment)
     v
   +-------------------+
   |  LiteLLM proxy    |  port 4000, transport + fallbacks + retries
   +-------------------+
     |                 \
     v                  v
  Ollama on mini     Ollama on MacBook Pro
  (small tier)       (large tier, LAN or Thunderbolt)

  Neo4j AuraDB (RAG graph)      Neon Postgres (LiteLLM state)
  SQLite on mini (traces, metrics, judge scores, eval runs)
```

### Request lifecycle

1. Authenticate and assign a request ID. Reject if no API key.
2. Normalize the conversation. Keep the last N turns (token budgeted), not only the last user message. This fixes the "show trace" follow up bug and gives multi turn context.
3. Classify with structured output. The classifier returns intent, complexity (small or large), needs_rag, enhanced_prompt, as a JSON schema enforced by Ollama's format parameter. Parse failures become impossible rather than handled.
4. Retrieve RAG context if needed, budgeted to a fixed token allowance.
5. Policy decides a deployment: tier from intent and complexity, then the best online device for that tier from the registry. If the preferred device is offline, degrade to the mini's fallback for that tier and mark the response as degraded in the trace.
6. Execute. Simple intents stream straight through. Complex code+reasoning goes through the decomposer, whose subtasks are scheduled across devices: one heavy subtask per device at a time, so two subtasks can run in parallel only when they land on different machines.
7. Validate every output deterministically. Judge the handoff outputs and the final answer. Retry once with feedback on failure, then degrade or drop with a reason.
8. Persist the trace, return the answer, and attach the trace block if the user asked for it.

## 4. Multi device design

### Device registry

A small in process service on the gateway that polls each configured Ollama node every 20 seconds: `GET /api/tags` for reachability and available models, `GET /api/ps` for loaded models and their memory footprint. It keeps a status per node: online or offline, models present, free memory estimate, and a rolling p50 latency. State changes are logged. A node that fails two polls in a row is marked offline; a node that answers one poll is marked online. The registry also exposes `GET /devices` so I can see what the gateway thinks is available.

### Policy engine

Pure function: (intent, complexity, registry snapshot, config) to a concrete LiteLLM deployment name and a device. Rules in order: pick the tier from intent and complexity; take the ordered candidate list for that tier from `jarvis.yaml`; skip candidates whose device is offline or whose model is not pulled; skip candidates whose device does not have enough free memory to load the model without swapping; return the first survivor. If nothing survives, return the mini's smallest model for that intent and flag the request as degraded. Because it is a pure function it is trivially unit testable with fake registry snapshots, which is the point.

### LiteLLM's role

LiteLLM keeps doing transport: one `model_name` per concrete deployment (for example `reasoning-large-mbp`, `reasoning-small-mini`), retries, timeouts, and a fallback map as a safety net for the window between a device dying and the registry noticing. Policy lives in the gateway, not in LiteLLM routing strategies, so there is exactly one place that decides where a request goes.

### Concurrency

One semaphore per device for the heavy tiers. The mini runs at most one 7B or 8B model call at a time plus the resident classifier and embedder. The MacBook Pro runs at most one large call at a time. The decomposer's parallel executor is re-enabled, but its worker count is the number of online devices, and each subtask acquires its device's semaphore. This is what makes parallelism safe again.

### Network

The MacBook Pro runs Ollama bound to `0.0.0.0:11434` (binding to a single interface IP was rejected because the service cannot start when that address is absent, which happens whenever the laptop changes network) and is reached by mDNS name. Ollama has no authentication, so a pf rule on the laptop allows port 11434 only from the mini's address while leaving loopback open: `block in quick on ! lo0 proto tcp from ! MINI_IP to any port 11434`, loaded from an anchor and re-enabled at boot by a LaunchDaemon running `pfctl -e -f /etc/pf.conf`. When the laptop is on the desk, a Thunderbolt bridge gives a private link and much lower latency, and the registry simply sees the node at a second address. When the laptop is away it is offline and the mini carries on. Tailscale is the option if I ever want the laptop reachable from elsewhere; it is not part of v2.

### Ollama settings on each node

`OLLAMA_MAX_LOADED_MODELS=2` on the mini (classifier plus one worker model), `1` on the laptop. `OLLAMA_NUM_PARALLEL=1` on both. `OLLAMA_KEEP_ALIVE` long for the classifier and embedder, short for the workers. `num_ctx` set explicitly per model in LiteLLM so context never silently overflows. Ollama runs as a launchd service on both machines.

## 5. Guardrails

### Operational

Every outbound call has a timeout and a wall clock guard. Every request has a total budget (default 10 minutes) and a token budget for input assembly: system prompt, conversation history, RAG context and handoff context each get an allowance and are truncated to fit before sending. A RAM watchdog on the gateway refuses to schedule a heavy call on a device when its free memory is below a threshold, which is how swap is prevented rather than detected after the fact. Per device semaphores as above.

### Output quality

`guards.py` holds the deterministic validators. Each returns a verdict and a reason: empty or too short; hit the length limit mid output; repetition ratio (repeated n grams) above threshold; non printable or mixed script character ratio; unbalanced code fences; JSON does not match schema; reasoning think/answer ratio indicates a spiral. These run on every model output and cost nothing.

The judge runs only at two boundaries: the reasoning output before it is handed to a code subtask, and the final answer before it is returned. It uses the general model with a fixed rubric and a JSON schema, returning a 1 to 5 score with a one line reason. Below threshold triggers one retry with the reason injected as feedback. A second failure drops the result and the reason is logged and surfaced in the trace. Judge scores are stored so I can see distributions over time.

### Input and output safety

The gateway requires an API key and CORS is restricted to Open WebUI's origin. RAG content is treated as untrusted: it is wrapped in a clearly delimited block with an instruction that it is reference material, and a lightweight injection detector flags retrieved chunks that contain instruction like phrases. A PII scrubber is designed into the outbound path so that if cloud models are ever added, nothing sensitive leaves the machine; in v2 it exists as a pass through with tests.

### Repo protections

Pre-commit with gitleaks and ruff. GitHub Actions runs ruff, pytest, and the eval suite against a mocked LiteLLM on every push. Branch protection on main requiring the workflow to pass. `.env.example` documents every variable; `config.py` is the only place environment is read.

## 6. Observability

Structured JSON logs with request ID on every line. A `traces` table in SQLite on the mini: request ID, timestamps per stage, intent, complexity, device and deployment chosen, degraded flag, tokens in and out, validator verdicts, judge scores, and the tracer's think/answer ratio for reasoning calls. The existing streaming tracer keeps writing thinking to disk but into a proper `traces/` directory with rotation, not `/tmp`. `GET /metrics` exposes Prometheus format counters and histograms (latency per stage, requests by tier and device, degraded count, judge score histogram). The trace block on demand stays, now generated from the persisted trace.

## 7. Evaluation

A `evals/golden.jsonl` of 30 or so prompts, each with the expected intent, expected complexity, whether RAG should fire, and a short rubric. `evals/run.py` executes them against the live gateway and reports routing accuracy, judge score distribution, and latency per tier. The same runner in CI uses a mocked LiteLLM so the routing and validation logic is regression tested without models. Results are written to the traces database and the latest summary table lives in the README. Every prompt or model change gets an eval run before it merges.

## 8. Configuration and secrets

`~/.jarvis.env` holds secrets only: Neo4j credentials, Neon connection string, LiteLLM master key, gateway API key. It is the only file with secrets and it is never in the repo. `jarvis.yaml` in the repo holds everything else: devices and their addresses, tier to deployment candidate lists, budgets, thresholds, model context sizes. `config.py` loads both with pydantic settings and validates on startup, failing fast with a readable message. The LiteLLM config references secrets as `os.environ/NAME`.

## 9. Repository layout

```
jarvis/
  README.md               architecture, quick start, eval results table
  pyproject.toml          uv managed, pinned
  jarvis.yaml             devices, tiers, budgets
  .env.example
  .pre-commit-config.yaml
  .github/workflows/ci.yml
  docs/
    adr/                  one file per decision, numbered
    architecture.md       this document, kept current
  src/jarvis/
    config.py
    gateway.py            FastAPI app, thin
    conversation.py       history normalization and token budgeting
    classify.py           structured output classifier
    rag.py                retrieval, watcher shares this
    registry.py           device registry
    policy.py             pure routing function
    llm.py                the one LiteLLM client, sync and streaming
    decomposer.py         LangGraph graph, device aware scheduling
    tracer.py             streaming tracer
    guards.py             deterministic validators, token budgeter
    judge.py              rubric judge with retry
    store.py              SQLite traces and metrics
    watcher.py            RAG watcher entry point
  evals/
    golden.jsonl
    run.py
  tests/
  scripts/
    start.sh  stop.sh  install-launchd.sh  setup-node.sh
```

## 10. Open decisions

Which large models go on the MacBook Pro: decided, see section 2. Remaining question is whether qwen3-coder:30b beats the instruct MoE on code by enough to justify a swap; the eval suite answers that later.

Thunderbolt bridge versus WiFi as the primary link. Thunderbolt is faster and private but only works at the desk; the registry handles both, so this is about which address is listed first.

Whether the MacBook Pro is still managed by an employer. If it has MDM or a usage policy, running a network service on it needs checking before we build on it.

Whether Open WebUI stays on the mini or moves to the laptop when present. Design says it stays on the mini so the entry point never moves.

## 11. Delivery plan

Week 1: publish safely. Secrets out, config.py, repo layout, pyproject, pre-commit, CI skeleton, README with architecture diagram, first ADRs (cloud free tiers over Docker, sequential over parallel on one device, policy in gateway not LiteLLM). Push to GitHub.

Week 2: reliability and guardrails on the mini alone. Non blocking gateway, conversation history, structured output classifier, guards.py, token budgeter, judge at boundaries, SQLite trace store, unit tests, golden set and eval runner. Fix the watcher path.

Week 3: multi device. Registry, policy engine, jarvis.yaml devices, MacBook Pro node setup script and launchd, per device semaphores, decomposer scheduling across devices, eval run comparing single device and dual device.

Week 4: hardening and polish. Metrics endpoint, launchd services on the mini, install from clean clone test, README eval table, ADRs for the routing and judge decisions, tag v2.0.0.

After v2: the layer split experiment (MLX distributed or llama.cpp RPC over Thunderbolt), written up as an experiment with measured tokens per second, not as a feature.

## 12. Decisions already made (become ADRs)

Cloud free tiers instead of local Docker for Neo4j and Postgres, because Docker cost 8 GB of RAM on the mini. Sequential subtask execution on a single device, parallel only across devices. A dedicated 3B classifier kept resident. Routing policy lives in the gateway; LiteLLM is transport. Deterministic validation on everything, LLM judge only at boundaries. The mini is always the entry point; the laptop is an optional worker.
