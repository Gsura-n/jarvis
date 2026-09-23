# ADR 0002: Subtasks execute sequentially on a single device; parallelism only across devices

Date: 2026-06 (recorded 2026-09-15)
Status: accepted

## Context

The task decomposer originally ran independent subtasks in parallel with a thread pool. On one 16 GB machine that meant two 7B or 8B models competing for memory, which caused Ollama to swap models in and out and produced timeouts and empty results.

## Decision

Execute subtasks in dependency order, one at a time per device. Parallelism is allowed only when subtasks are scheduled on different devices, each device guarded by its own semaphore (see the v2 design, section 4).

## Consequences

Slower wall clock for large decompositions on a single machine, but stable. The parallel executor is kept in the code (`execute_tasks_parallel`) for the multi device phase rather than deleted.
