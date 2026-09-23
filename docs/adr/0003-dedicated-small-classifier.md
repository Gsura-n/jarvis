# ADR 0003: A dedicated resident 3B model for classification

Date: 2026-06 (recorded 2026-09-15)
Status: accepted

## Context

Every request is classified for intent, RAG need and (in v2) complexity before any work happens. Using an 8B model for this added seconds of latency to every request and forced a model swap when the worker model was different.

## Decision

Use llama3.2:3b as a resident classifier on the mini with a long keep alive, so classification is fast and never evicts the worker model.

## Consequences

Classification quality at 3B is adequate for the intent taxonomy but JSON output is unreliable when parsed from free text. v2 enforces a JSON schema through Ollama's structured output so parse failures cannot happen; the keyword fallback in the classifier becomes dead code once that lands.
