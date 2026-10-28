# Jules Scan Report (Oct 28, 2026)

## Baseline

```
........................................................................ [ 36%]
........................................................................ [ 73%]
....................................................                     [100%]
196 passed in 7.46s

All checks passed!
```
The baseline remains unchanged at 196 passing tests and a clean ruff run.

## Findings

| Severity | Category | File | Description | Scenario |
|----------|----------|------|-------------|----------|
| High | Correctness | `src/jarvis/gateway.py:560` | Async gateway is blocked by synchronous `.result()` calls in thread pool. | The `completions` endpoint uses `classify_future.result()` (and `rag_future.result()`) directly inside `with ThreadPoolExecutor`. Because the endpoint is `async def`, these blocking calls will stall the FastAPI event loop for up to the classifier's timeout (15s) and Neo4j query duration, starving other concurrent API requests. |
| High | Correctness | `src/jarvis/task_decomposer.py:439, 449, 454, 507` | Wipeout during synthesis due to unchecked `call_model` errors. | Although `aggregate_results` correctly filters broken subtasks via `is_broken_result`, the `smart_compress` function and the progressive synthesis loop call `call_model` without checking for the `[ERROR] ...` return string. If the summarization or synthesis call fails, the accumulated synthesis is wiped out and replaced with the error message. |
| High | Correctness | `src/jarvis/task_decomposer.py:342` | Missing validation for DAG cycle in `execute_tasks`. | If the subtask dependencies form a cycle, `ready` will be empty, and the loop breaks early, silently returning partial results without signaling an error. |
| Medium | Security | `src/jarvis/gateway.py:142` | Missing timeout on Neo4j RAG query. | The `rag_query` function executes a Neo4j query via `session.run()` without any timeout. A slow or hung database connection will stall the request indefinitely. |
| Medium | Correctness | `src/jarvis/policy.py:81` | `decide` can return an empty deployment for an unknown tier/empty fallback. | If a tier has no configured candidates, `walk_tier` returns a `Decision` with `deployment=""`. This will be passed to LiteLLM, resulting in an error due to missing/empty model. |
| Low | Config Drift | `jarvis.yaml:22,34` | `min_free_gb_for_heavy` is configured but never read by the code. | The config has keys `min_free_gb_for_heavy` for devices, but `grep` shows they are never accessed in `src/jarvis/`. |
| Low | Config Drift | `jarvis.yaml:75` | `request_wall_s` is configured but never read or enforced by the code. | The config has key `budgets.request_wall_s`, but it is never accessed in `src/jarvis/`. |

## Top 3 Fixes

1. **Unblock the Gateway Event Loop**: The `ThreadPoolExecutor` `.result()` calls in `src/jarvis/gateway.py` are a critical bottleneck. They should be awaited using `asyncio.to_thread` or `asyncio.wrap_future` so that FastAPI can continue serving other concurrent requests while the intent classifier and RAG queries run.
2. **Add Timeout to Neo4j Query**: The `rag_query` function in `src/jarvis/gateway.py` needs a strict timeout. Without it, a hung database connection can freeze requests indefinitely, compounding the blocking issue above.
3. **Handle Errors During Synthesis**: The `smart_compress` and progressive synthesis loop in `src/jarvis/task_decomposer.py` must handle `[ERROR]` responses from `call_model`. They should retry or fail gracefully instead of blindly accumulating the error string and wiping out the final answer.
