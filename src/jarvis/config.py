"""
The single place where Jarvis reads its environment.

Secrets come from ~/.jarvis.env (override the path with JARVIS_ENV_FILE).
Topology, tiers and budgets come from jarvis.yaml at the repo root
(override with JARVIS_CONFIG). Nothing else in the package calls
os.getenv or load_dotenv directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> Path:
    env_file = Path(os.getenv("JARVIS_ENV_FILE", str(Path.home() / ".jarvis.env"))).expanduser()
    if env_file.exists():
        load_dotenv(env_file, override=True)
    return env_file


def _load_yaml() -> tuple[dict, Path]:
    cfg = Path(os.getenv("JARVIS_CONFIG", str(REPO_ROOT / "jarvis.yaml"))).expanduser()
    data: dict = {}
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
    return data, cfg


def _expand(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class Settings:
    env_file: Path
    config_file: Path

    # LiteLLM proxy
    litellm_base: str
    litellm_key: str

    # Neo4j (RAG graph)
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Ollama on this machine (embeddings, health)
    ollama_base: str
    embed_model: str

    # Paths
    log_dir: Path
    trace_dir: Path
    rag_cache_file: Path

    # RAG projects to watch: list of {name, path}
    rag_projects: list

    # Everything from jarvis.yaml, for the registry/policy layers
    topology: dict

    def missing_required(self) -> list[str]:
        required = {
            "LITELLM_MASTER_KEY": self.litellm_key,
            "NEO4J_URI": self.neo4j_uri,
            "NEO4J_USER": self.neo4j_user,
            "NEO4J_PASSWORD": self.neo4j_password,
        }
        return [k for k, v in required.items() if not v]


def load_settings() -> Settings:
    env_file = _load_env()
    topology, config_file = _load_yaml()
    paths = topology.get("paths", {})
    rag = topology.get("rag", {})

    log_dir = _expand(paths.get("log_dir", ".logs"))
    trace_dir = _expand(paths.get("trace_dir", log_dir / "traces"))

    projects = []
    for p in rag.get("projects", []):
        projects.append({"name": p["name"], "path": str(Path(p["path"]).expanduser())})

    return Settings(
        env_file=env_file,
        config_file=config_file,
        litellm_base=os.getenv("LITELLM_BASE", "http://localhost:4000/v1"),
        # LITELLM_MASTER_KEY is the canonical name; LITELLM_KEY kept for older env files
        litellm_key=os.getenv("LITELLM_MASTER_KEY", os.getenv("LITELLM_KEY", "")),
        neo4j_uri=os.getenv("NEO4J_URI", ""),
        neo4j_user=os.getenv("NEO4J_USER", ""),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        ollama_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        log_dir=log_dir,
        trace_dir=trace_dir,
        rag_cache_file=_expand(rag.get("cache_file", "~/.jarvis_rag_cache.json")),
        rag_projects=projects,
        topology=topology,
    )


settings = load_settings()
