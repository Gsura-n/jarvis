# ADR 0007: Large model first for design and build requests; decomposer as fallback

**Status:** Proposed
**Date:** 2026-09-23
**Deciders:** Gauttam

## Context

The LangGraph decomposer exists because 8B models on a 16 GB mini could not handle "design and build" requests in one call. A decomposed request costs a complexity call and a split call on llama3.1:8b, then 3 to 6 subtask calls across deepseek-r1:8b and qwen2.5-coder:7b, then compression and synthesis calls. With `OLLAMA_MAX_LOADED_MODELS=2` the mini reloads models several times per request. Each step is also a failure point (see the synthesis and dependency bugs in the Sept 23 review).

The MacBook Pro now serves qwen3:30b-a3b at 39 tok/s with one model resident. That changes the premise the decomposer was built on.

## Decision

When the large tier is online, a `code+reasoning` request goes to one call on the 30B model with a prompt that asks for a short design followed by the implementation. The decomposer runs only when the large tier is offline, or when the client explicitly asks for it (model name `jarvis-decompose`). This is provisional until the golden eval compares the two.

## Options Considered

### Option A: Always decompose (current)
| Dimension | Assessment |
|-----------|------------|
| Complexity | High: 6 to 10 model calls, 3 or 4 different models |
| Cost | Minutes of latency, frequent model reloads |
| Scalability | Parallelism helps only across devices |
| Familiarity | Built and debugged already |

**Pros:** Interesting orchestration to show; works on the mini alone.
**Cons:** Most calls and most failure points; quality bounded by 8B models and a 4K default context.

### Option B: Large model first, decomposer as fallback (recommended)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low on the main path |
| Cost | One call at ~39 tok/s, no reloads on the laptop |
| Scalability | Mini stays free for classify, embed and small tier |
| Familiarity | Reuses existing pieces |

**Pros:** Fewer moving parts on the common path; decomposer still demonstrable and still covers the laptop being away.
**Cons:** The showcase feature runs less often. Needs the eval to prove the choice rather than assert it.

### Option C: Remove the decomposer
**Pros:** Least code. **Cons:** No answer for the laptop being away; loses a strong interview talking point.

## Trade-off Analysis

The question is empirical: does splitting a task across small models beat one call to a larger model? The evidence available (39 tok/s, single resident model, no swap) points to B for both latency and quality, but it has not been measured. B is chosen because it is cheap to reverse: it is a policy rule, not a rewrite. Decision rule for revisiting: if the decomposed path beats the single large call by 0.5 or more in mean judge score on the design and build slice of the golden set at under 2x the latency, flip the default.

## Consequences

- Easier: the common path is one call; timeouts and cascades mostly disappear.
- Harder: two execution modes to keep tested.
- Revisit: after the first eval run, and again if qwen3-coder:30b replaces the instruct model.

## Action Items

1. [ ] Policy rule in `policy.py`: large tier online means single call.
2. [ ] Fix decomposer bugs before it becomes the fallback: synthesis overwrite on error, silent dependency stall, unenforced request budget.
3. [ ] Golden set gets at least 8 design and build prompts; eval runs both modes.
4. [ ] Put the comparison table in the README whatever it shows.
