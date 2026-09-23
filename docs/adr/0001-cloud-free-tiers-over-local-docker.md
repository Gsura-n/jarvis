# ADR 0001: Cloud free tiers instead of local Docker for Neo4j and Postgres

Date: 2026-06 (recorded 2026-09-15)
Status: accepted

## Context

The first Jarvis stack ran Neo4j and Postgres in Docker on the Mac mini (16 GB unified memory). Docker Desktop plus the two databases used about 8 GB of RAM and pushed the machine into swap whenever a model was loaded.

## Decision

Move the graph to Neo4j AuraDB Free and LiteLLM's state store to Neon Postgres Free. Run everything else natively (no Docker on the mini).

## Consequences

Freed roughly 5 GB of RAM and eliminated swap. The RAG graph and LiteLLM state now depend on an internet connection and on two free tiers with usage caps. Both can be swapped back to local services on a machine with more memory; the code talks to them only through connection strings in ~/.jarvis.env.
