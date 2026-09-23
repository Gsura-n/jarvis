# ADR 0008: Classifier returns labels only; explicit model choice skips it

**Status:** Proposed
**Date:** 2026-09-23
**Deciders:** Gauttam

## Context

Every request pays for llama3.2:3b to both rewrite the prompt and classify it. The downstream model then sees only the rewritten prompt, never the user's words. A 3B rewrite can drop constraints or shift the ask, and producing it (100 to 300 output tokens) is most of the classification latency. Parsing is by string cleanup, which is the source of the known JSON parse failures. The classifier sees only the last user message, which causes the "show trace" follow up bug. Separately, `/v1/models` advertises `coding`, `reasoning` and `general`, but the gateway ignores the `model` field, so picking a model in Open WebUI changes nothing.

## Decision

1. The classifier returns only `{intent, complexity, needs_rag}`, enforced with Ollama's `format` JSON schema, and sees the last few turns of conversation. No prompt rewrite; the original messages go downstream.
2. `/v1/models` lists `jarvis` (auto routed) plus explicit tiers. If the request names an explicit tier, classification is skipped.
3. Trace requests are detected from the conversation, not a keyword list, or exposed as an explicit `jarvis-trace` model or header.

## Options Considered

### Option A: Rewrite and classify with the 3B (current)
**Pros:** Short, vague prompts get expanded. **Cons:** Lossy, slow, unparseable output, no history.

### Option B: Labels only with schema enforced output (recommended)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | A few output tokens instead of hundreds |
| Reliability | Parse failures become impossible |
| Familiarity | Same model, smaller prompt |

**Pros:** Faster, no information loss, deterministic parsing.
**Cons:** Loses prompt expansion (which the 30B model does not need).

### Option C: Embedding classifier, no LLM call
Embed the request with the resident nomic-embed-text and pick the nearest labeled examples.
**Pros:** Tens of milliseconds; the embedding is reused for RAG. **Cons:** Needs a labeled example set; weaker on unusual phrasing.

## Trade-off Analysis

B fixes the known bugs with the least new code. C could remove the LLM from the routing path entirely and is a good experiment once the golden set exists, since routing accuracy is exactly what the golden set measures. Try C against B on the same set before choosing between them long term.

## Consequences

- Easier: routing accuracy becomes a single number; latency drops on every request.
- Harder: short prompts are passed through as is.
- Revisit: after the golden set exists, run C against B.

## Action Items

1. [ ] `classify.py` with a JSON schema and last N turns.
2. [ ] Honor `body["model"]` in the gateway.
3. [ ] Remove `ENHANCE_SYSTEM` rewrite logic and keyword fallback.
