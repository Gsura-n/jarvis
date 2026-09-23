# Jarvis

A local AI orchestration gateway. One OpenAI compatible endpoint that classifies each request, decides how much model it deserves, runs it on whichever of two Macs is best placed to serve it, and returns an answer that has passed a set of quality gates. Everything runs on hardware I own.

Status: v2 in progress. The v1 pipeline works end to end on the Mac mini; v2 makes it reliable, measurable, secure and multi device. See [docs/architecture.md](docs/architecture.md) for the design and [docs/adr/](docs/adr/) for the decisions.

## How it works

```
Open WebUI / Continue / curl
            |
            v
   +-------------------+
   |  Jarvis Gateway   |  :4001  enhance -> classify -> RAG -> route -> execute -> validate
   |  (FastAPI, mini)  |
   +-------------------+
     |      |       |
     |      |       +--> Device Registry   which nodes are online, what is loaded, free memory
     |      +----------> Policy Engine     tier x availability -> concrete deployment
     v
   +-------------------+
   |  LiteLLM proxy    |  :4000  transport, retries, fallbacks
   +-------------------+
     |                 \
     v                  v
  Ollama on Mac mini   Ollama on MacBook Pro
  small tier           large tier (one resident 30B MoE)

  Neo4j AuraDB (RAG graph)   Neon Postgres (LiteLLM state)   SQLite (traces, evals)
```

Requests are classified by a resident 3B model into an intent (general, code, reasoning, code+reasoning) and, in v2, a complexity (small or large). Simple requests stream straight through the matching small model on the mini. Large or design-and-build requests go to the 30B model on the MacBook Pro when it is online, and to the mini's smaller models when it is not. Design-and-build requests are decomposed into subtasks by a LangGraph graph with structured handoff from a reasoning step to a code step, cascade protection for failed subtasks, and a memory safe tracer that streams model thinking to disk instead of holding it in RAM.

## Hardware

| Machine | Role | Models |
|---|---|---|
| Mac mini M4, 16 GB | control plane, always on | llama3.2:3b (classifier), nomic-embed-text, llama3.1:8b, qwen2.5-coder:7b, deepseek-r1:8b |
| MacBook Pro M1 Pro, 32 GB | worker, sometimes away | qwen3:30b-a3b-instruct-2507-q4_K_M (18 GB, 39 tok/s), qwen2.5-coder:14b (benchmark only) |

## Quick start (Mac mini)

Requirements: macOS on Apple silicon, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) with the small tier models pulled, and free tier accounts on Neo4j AuraDB and Neon.

```bash
git clone https://github.com/Gsura-n/jarvis.git ~/projects/jarvis
cd ~/projects/jarvis
uv sync --all-groups
uv run pre-commit install

cp .env.example ~/.jarvis.env    # then fill in every value
./scripts/start.sh               # LiteLLM :4000, gateway :4001, RAG watcher, Open WebUI :8080 if installed
./scripts/stop.sh
```

Point any OpenAI compatible client at `http://localhost:4001/v1`. Ask "which model answered?" or "show trace" at the end of a question to get the routing and timing trace appended to the answer.

Logs and traces land in `.logs/` (gitignored). Topology, tiers and budgets are in `jarvis.yaml`; secrets are only ever in `~/.jarvis.env`.

## Repository layout

```
src/jarvis/          the package (gateway, decomposer, tracer, RAG watcher, config)
configs/litellm.yaml LiteLLM deployments; secrets referenced as os.environ/NAME
jarvis.yaml          devices, tiers, budgets, guard thresholds
scripts/             start.sh, stop.sh
tests/               pure function tests (no models, no network, no secrets)
evals/               golden set and runner (week 2)
docs/architecture.md the design, kept current
docs/adr/            one file per decision
```

## Development

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
uv run pre-commit run --all-files
```

CI runs gitleaks, ruff and pytest on every push and pull request.

## Roadmap

Week 1: publish safely (this commit). Week 2: non blocking gateway, conversation history, structured output classifier, deterministic guards, judge at boundaries, SQLite traces, golden set and evals. Week 3: device registry, policy engine, parallel subtasks across devices. Week 4: metrics endpoint, launchd services, clean clone install test, v2.0.0. After that: pipeline parallel experiments splitting one model across both machines, written up with measured numbers.

## Evidence

This table is filled in by the eval suite from week 2 on.

| Measurement | Value |
|---|---|
| Routing accuracy on golden set | pending |
| Judge agreement with human labels | pending |
| Same prompt, temperature 0, 10 runs, identical outputs | pending |
| Single device vs dual device wall clock on decomposed tasks | pending |

## License

MIT
