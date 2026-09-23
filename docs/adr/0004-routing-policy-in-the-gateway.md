# ADR 0004: Routing policy lives in the gateway; LiteLLM is transport

Date: 2026-09-15
Status: accepted

## Context

LiteLLM offers routing strategies, fallbacks and health checks. The gateway also needs to decide where a request runs, based on intent, complexity and which devices are online with memory headroom. Putting device logic in both places would make it impossible to explain why a request ran where it did.

## Decision

The gateway owns policy: a pure function from (tier, registry snapshot, config) to a concrete LiteLLM deployment name. LiteLLM gets one deployment per concrete model on a concrete device, plus timeouts, retries and a fallback map used only as a safety net for the seconds between a device dying and the registry noticing.

## Consequences

One place explains every routing decision, and it is unit testable with fake registry snapshots. LiteLLM's own routing strategies stay at simple-shuffle and are not relied on.
