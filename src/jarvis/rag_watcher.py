"""
rag_watcher.py
Watches project folders for file changes and incrementally updates Neo4j AuraDB.
Only re-ingests changed/new files — not the full project.
Supports multiple projects simultaneously.
"""

import os
import sys
import time
import json
import hashlib
import requests
from pathlib import Path
from jarvis.config import settings
from neo4j import GraphDatabase
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = settings.neo4j_uri
NEO4J_USER     = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password
OLLAMA_BASE    = settings.ollama_base
EMBED_MODEL    = settings.embed_model

# File types to watch
EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".py", ".cls", ".trigger", ".apex"}

# Directories to skip
SKIP_DIRS = {"node_modules", ".next", ".git", "dist", "build", ".venv", "__pycache__"}

# Hash cache file — tracks file hashes to detect real changes (path set in jarvis.yaml)
HASH_CACHE_FILE = str(settings.rag_cache_file)

# Projects to watch — configured under rag.projects in jarvis.yaml
PROJECTS = settings.rag_projects

# ── Neo4j ─────────────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Hash cache ────────────────────────────────────────────────────────────────
def load_hash_cache() -> dict:
    try:
        with open(HASH_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_hash_cache(cache: dict):
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


# ── Embedding ─────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text[:2000]},
            timeout=15,
        )
        return r.json().get("embeddings", [[]])[0]
    except Exception:
        return []


# ── Neo4j operations ──────────────────────────────────────────────────────────
def upsert_file(file_path: Path, project_name: str, project_root: Path):
    """Ingest or update a single file in Neo4j."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        relative = str(file_path.relative_to(project_root))
        embedding = get_embedding(content)

        if not embedding:
            print(f"  ⚠️  No embedding for {relative}")
            return

        with driver.session() as session:
            session.run("""
                MERGE (f:File {path: $path, repo: $repo})
                SET f.name = $name,
                    f.extension = $ext,
                    f.content_preview = $preview,
                    f.embedding = $embedding,
                    f.repo = $repo,
                    f.updated_at = datetime()
            """,
                path=relative,
                name=file_path.name,
                ext=file_path.suffix,
                preview=content[:500],
                embedding=embedding,
                repo=project_name,
            )

        print(f"  ✅ Upserted: {relative}")

    except Exception as e:
        print(f"  ❌ Error upserting {file_path}: {e}")


def delete_file(relative_path: str, project_name: str):
    """Remove a deleted file from Neo4j."""
    try:
        with driver.session() as session:
            session.run("""
                MATCH (f:File {path: $path, repo: $repo})
                DETACH DELETE f
            """, path=relative_path, repo=project_name)
        print(f"  🗑️  Deleted: {relative_path}")
    except Exception as e:
        print(f"  ❌ Error deleting {relative_path}: {e}")


# ── File event handler ────────────────────────────────────────────────────────
class RagEventHandler(FileSystemEventHandler):
    def __init__(self, project_name: str, project_root: Path):
        self.project_name = project_name
        self.project_root = project_root
        self.hash_cache = load_hash_cache()
        self._debounce = {}  # path -> last event time

    def _should_process(self, path: str) -> bool:
        """Check extension and skip dirs."""
        p = Path(path)
        if p.suffix not in EXTENSIONS:
            return False
        if any(skip in p.parts for skip in SKIP_DIRS):
            return False
        return True

    def _debounced(self, path: str, delay: float = 1.0) -> bool:
        """Return True if enough time has passed since last event for this path."""
        now = time.time()
        last = self._debounce.get(path, 0)
        if now - last < delay:
            return False
        self._debounce[path] = now
        return True

    def _process_change(self, path: str):
        if not self._should_process(path):
            return
        if not self._debounced(path):
            return

        p = Path(path)
        if not p.exists():
            return

        # Check if content actually changed
        current_hash = file_hash(path)
        if self.hash_cache.get(path) == current_hash:
            return  # No real change

        print(f"\n📝 Change detected: {p.name} ({self.project_name})")
        self.hash_cache[path] = current_hash
        save_hash_cache(self.hash_cache)
        upsert_file(p, self.project_name, self.project_root)

    def _process_delete(self, path: str):
        if not self._should_process(path):
            return

        p = Path(path)
        try:
            relative = str(p.relative_to(self.project_root))
        except ValueError:
            return

        print(f"\n🗑️  Deleted: {p.name} ({self.project_name})")
        if path in self.hash_cache:
            del self.hash_cache[path]
            save_hash_cache(self.hash_cache)
        delete_file(relative, self.project_name)

    def on_modified(self, event):
        if not event.is_directory:
            self._process_change(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._process_change(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._process_delete(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._process_delete(event.src_path)
            self._process_change(event.dest_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🔍 Jarvis RAG Watcher starting...")
    print(f"   Watching {len(PROJECTS)} project(s)")
    print(f"   Extensions: {', '.join(sorted(EXTENSIONS))}")
    print(f"   Hash cache: {HASH_CACHE_FILE}")
    print()

    observer = Observer()

    for project in PROJECTS:
        name = project["name"]
        path = Path(project["path"])

        if not path.exists():
            print(f"  ⚠️  Project path not found, skipping: {path}")
            continue

        handler = RagEventHandler(name, path)
        observer.schedule(handler, str(path), recursive=True)
        print(f"  👁️  Watching: {name}")
        print(f"       Path: {path}")

    print()
    print("✅ Watcher active — save a file to trigger incremental ingest")
    print("   Press Ctrl+C to stop\n")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping watcher...")
        observer.stop()

    observer.join()
    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
