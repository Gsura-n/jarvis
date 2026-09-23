# ADR 0005: The MacBook Pro serves exactly one large model

Date: 2026-09-15
Status: accepted

## Context

The MacBook Pro (M1 Pro, 32 GB, GPU wired limit raised to 26 GB) measured qwen3:30b-a3b-instruct-2507-q4_K_M at 18 GB and 39 tok/s, and qwen2.5-coder:14b at 9 GB and 12 tok/s. The two cannot be resident together, and a swap costs about 10 seconds. The decomposer alternates reasoning and code subtasks, which would trigger that swap constantly.

## Decision

The 30B MoE is the only model in routine routing on that node and serves both reasoning-large and code-large. qwen2.5-coder:14b stays pulled for benchmarks and emergency fallback only. Small code work goes to the mini's qwen2.5-coder:7b. qwen3.6:35b-a3b was rejected because its smallest quantization is 23 GB.

## Consequences

No swap thrash on the laptop and a simpler policy engine. Code quality on the large tier depends on the MoE; qwen3-coder:30b is to be benchmarked through the eval suite and could replace it if it wins by a clear margin.
