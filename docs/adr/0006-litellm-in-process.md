# ADR 0006: Run LiteLLM in process instead of as a proxy

**Status:** Proposed
**Date:** 2026-09-23
**Deciders:** Gauttam

## Context

ADR 0004 moved routing policy into the gateway. Since then the LiteLLM proxy on port 4000 only does transport: forward a request to one Ollama node, retry, time out. To do that it needs its own process, a Neon Postgres database (used for almost nothing, since spend logs are disabled and there is one master key), a master key secret, and a second config file whose deployment names must stay in sync with `jarvis.yaml` by hand. `start.sh` sleeps 8 seconds waiting for it. Every model call makes an extra local HTTP hop.

## Decision

Replace the proxy with the LiteLLM Python library inside the gateway (`litellm.Router` built from `jarvis.yaml` at startup, called with `acompletion`). One module, `llm.py`, owns it. `configs/litellm.yaml`, `DATABASE_URL` and `LITELLM_MASTER_KEY` go away. Clients (Open WebUI, Continue) point only at the gateway on 4001.

## Options Considered

### Option A: Keep the proxy (current)
| Dimension | Assessment |
|-----------|------------|
| Complexity | High: two processes, two configs, a cloud DB |
| Cost | One extra secret to rotate, Neon dependency |
| Scalability | Fine, but unneeded for one user |
| Familiarity | Already running |

**Pros:** LiteLLM admin UI; clients can bypass the gateway and hit models directly.
**Cons:** Config duplication, extra hop, startup ordering, Postgres for no real purpose.

### Option B: LiteLLM Router in process (recommended)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low: one process, config generated from `jarvis.yaml` |
| Cost | Removes Neon and the master key |
| Scalability | Same as A for this workload |
| Familiarity | Same library, different entry point |

**Pros:** Keeps retries, fallbacks, and the provider abstraction (cloud fallback later is a config change). `jarvis.yaml` becomes the single source of truth. Native async.
**Cons:** Lose the LiteLLM UI. Heavier import than a plain HTTP client.

### Option C: Plain httpx to Ollama's OpenAI endpoint
| Dimension | Assessment |
|-----------|------------|
| Complexity | Lowest |
| Cost | No dependencies |
| Scalability | Same |
| Familiarity | Needs retry and fallback code written by hand |

**Pros:** Smallest surface, easiest to read line by line (good for Python fluency practice).
**Cons:** Reimplements what LiteLLM already does; adding a cloud provider later means new code.

## Trade-off Analysis

A buys a UI and direct model access at the cost of a database, a secret and a duplicated config. Neither benefit matters for a single user system where every request should pass through the gateway anyway. B keeps everything LiteLLM is good at and deletes everything it was costing. C is attractive for learning but gives up the provider abstraction that the non goals section explicitly wants to keep open.

## Consequences

- Easier: one config, one process, faster startup, no Neon password to rotate before publishing.
- Harder: nothing can reach models except through the gateway (this is intended).
- Revisit: if Jarvis ever becomes multi user, the proxy's virtual keys and spend tracking come back into play.

## Action Items

1. [ ] `llm.py`: build `litellm.Router` from `jarvis.yaml` tiers, with `num_ctx` per deployment.
2. [ ] Replace the four hand written HTTP helpers (gateway `chat`, `chat_stream`, decomposer `call_model`, tracer) with `llm.py`.
3. [ ] Delete `configs/litellm.yaml`, drop `DATABASE_URL` and `LITELLM_MASTER_KEY` from `.env.example` and `start.sh`.
4. [ ] Repoint Continue at the gateway.
