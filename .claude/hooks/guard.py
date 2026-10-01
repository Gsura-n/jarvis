#!/usr/bin/env python3
"""Golden Rules guard for Claude Code (PreToolUse hook).

Reads the tool call as JSON on stdin and answers with a permission decision:
  deny  -> the call is blocked and the reason is shown to the agent
  ask   -> a human must approve the call
  (no output, exit 0) -> normal permission flow

Rule IDs refer to RULES.md in the golden-rules repo.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

# Files an agent must not change on its own (AGT-003). A human can still
# approve the change when asked.
PROTECTED_PATHS = (
    ".claude/",
    ".codex/",
    ".grok/",
    ".cursor/",
    ".github/workflows/",
    ".github/CODEOWNERS",
    ".pre-commit-config.yaml",
    "scripts/golden/",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".mcp.json",
    ".cursorrules",
    ".github/copilot-instructions.md",
)

SECRET_FILE = re.compile(
    r"(^|/)(\.env(\.[\w.-]+)?|id_rsa[\w.-]*|id_ed25519[\w.-]*|[\w.-]+\.pem|[\w.-]+\.key|"
    r"credentials\.json|\.aws/credentials|\.netrc|\.npmrc|\.pypirc|secrets?\.(ya?ml|json|toml))$"
)
SECRET_OK = re.compile(r"\.(example|sample|template|dist)$")

BYPASS_FLAGS = (
    "--dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    "--yolo",
    "--no-verify",
)

INSTALL_PATTERNS = [
    # (regex on a single command segment, allow-if regex)
    (
        r"^(pip3?|python3? -m pip|uv pip) install\b",
        r"install(\s+-[^\s]+)*\s+(-r|--requirement|-e|--editable)\b|install\s+\.(\[[\w,]+\])?(\s|$)|install\s+(-U\s+|--upgrade\s+)?pip$",
    ),
    (r"^uv add\b", None),
    (r"^poetry add\b", None),
    (r"^(npm|pnpm) (install|i|add)\s+(?!-)[^\s]", None),
    (r"^yarn add\b", None),
    (r"^bun add\b", None),
    (r"^cargo add\b", None),
    (r"^go (get|install)\b", None),
    (r"^gem install\b", None),
    (r"^brew install\b", None),
]

ASK_COMMANDS = [
    (r"^git push\b", "Pushing changes off this machine needs a human (OPS-001)."),
    (r"^git reset --hard\b", "Hard reset discards work (AGT-006)."),
    (r"^git (rebase|commit --amend)\b", "History rewrite needs a human (AGT-006)."),
    (r"^git clean\b.*-[a-zA-Z]*f", "git clean deletes untracked files (AGT-006)."),
    (
        r"^(terraform|tofu|terragrunt) (apply|destroy|import|state)\b",
        "Infrastructure changes need a human (AGT-006).",
    ),
    (
        r"^pulumi (up|destroy|import)\b",
        "Infrastructure changes need a human (AGT-006).",
    ),
    (
        r"^kubectl (apply|delete|create|replace|patch|scale|drain|edit)\b",
        "Cluster changes need a human (AGT-006).",
    ),
    (
        r"^helm (install|upgrade|uninstall|rollback)\b",
        "Cluster changes need a human (AGT-006).",
    ),
    (
        r"^aws\b.*\b(delete|terminate|remove|put-bucket-policy|create-access-key)\b",
        "Cloud changes need a human (AGT-006).",
    ),
    (r"^gcloud\b.*\b(delete|remove)\b", "Cloud changes need a human (AGT-006)."),
    (r"^az\b.*\b(delete|remove)\b", "Cloud changes need a human (AGT-006)."),
    (
        r"\b(drop\s+(table|database|schema)|truncate\s+table)\b",
        "Destructive SQL needs a human (AGT-006).",
    ),
    (
        r"\b(alembic|prisma|flyway|liquibase|knex)\b.*\b(upgrade|migrate|deploy|downgrade)\b",
        "Migrations against a database need a human (AGT-006).",
    ),
    (
        r"^(gh|hub) (pr merge|release create|repo delete)\b",
        "Merging, releasing or deleting needs a human (GOV-003).",
    ),
    (r"^docker (push|system prune)\b", "Publishing or pruning images needs a human."),
    (
        r"^(npm|pnpm|yarn) publish\b|^twine upload\b|^cargo publish\b",
        "Publishing packages needs a human.",
    ),
]


def decide(kind: str, reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": kind,
                    "permissionDecisionReason": f"[golden-rules] {reason}",
                }
            }
        )
    )
    sys.exit(0)


def project_dir() -> str:
    return os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def rel_to_project(path: str, cwd: str) -> str | None:
    """Return path relative to the project, or None when it is outside."""
    root = project_dir()
    full = os.path.realpath(os.path.join(cwd, os.path.expanduser(path)))
    if full == root:
        return ""
    if full.startswith(root + os.sep):
        return os.path.relpath(full, root)
    return None


def is_protected(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    for p in PROTECTED_PATHS:
        if p.endswith("/"):
            if rel.startswith(p) or ("/" + p) in ("/" + rel):
                return True
        elif rel == p or rel.endswith("/" + p):
            return True
    return False


def is_secret_file(path: str) -> bool:
    name = path.replace(os.sep, "/")
    return bool(SECRET_FILE.search(name)) and not SECRET_OK.search(name)


def segments(command: str) -> list[str]:
    # Split on shell control operators. Good enough for policy checks; it does
    # not need to be a full shell parser because deny rules also scan the raw text.
    parts = re.split(r"\|\||&&|;|\||\n|\$\(|`", command)
    out = []
    for p in parts:
        p = p.strip().lstrip("(").strip()
        # drop leading env assignments and sudo / command wrappers
        while True:
            m = re.match(
                r"^([A-Za-z_][A-Za-z0-9_]*=\S*\s+|sudo\s+|command\s+|exec\s+|time\s+|nohup\s+)",
                p,
            )
            if not m:
                break
            p = p[m.end() :]
        if p:
            out.append(p)
    return out


def check_bash(command: str, cwd: str) -> None:
    raw = command

    for flag in BYPASS_FLAGS:
        if re.search(rf"(^|\s){re.escape(flag)}(\s|=|$)", raw):
            decide(
                "deny",
                f"'{flag}' skips safety checks. Run the command without it (AGT-002).",
            )

    if re.search(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", raw):
        decide("deny", "Piping a download into a shell runs unreviewed code (SUP-001).")

    if re.search(
        r"\bgit\s+push\b.*(\s--force(-with-lease)?\b|\s-f\b|\s--mirror\b|\s--delete\b|\s\+\S|\s:\S)",
        raw,
    ):
        decide(
            "deny",
            "Force pushes, mirror pushes and remote deletes are not allowed (AGT-006).",
        )

    if re.search(r"\bgit\s+(filter-branch|filter-repo)\b", raw):
        decide("deny", "History rewrites are not allowed (AGT-006).")

    if re.search(r"\b(printenv|env)\s*($|\|)", raw):
        decide("deny", "Dumping the environment can expose secrets (SEC-006).")

    for seg in segments(raw):
        try:
            words = shlex.split(seg, posix=True)
        except ValueError:
            words = seg.split()
        if not words:
            continue

        # Reading or touching secret files through the shell (SEC-006).
        for w in words[1:]:
            if not w.startswith("-") and is_secret_file(w):
                decide(
                    "deny",
                    f"'{w}' looks like a secret file. Read secrets at runtime from the secrets manager, never into the session (SEC-006).",
                )

        # Recursive deletes (AGT-006).
        if words[0] == "rm" and any(
            re.match(r"^-[a-zA-Z]*[rR]", w) or w == "--recursive" for w in words[1:]
        ):
            targets = [w for w in words[1:] if not w.startswith("-")]
            for t in targets:
                if t in ("/", "~", "~/", "*", ".", "..") or rel_to_project(t, cwd) in (
                    None,
                    "",
                ):
                    decide(
                        "deny",
                        f"Recursive delete of '{t}' reaches outside the project or the whole project (AGT-006).",
                    )
            decide(
                "ask",
                "Recursive delete inside the project. Confirm the paths are correct (AGT-006).",
            )

        # Writes to protected files through the shell (AGT-003).
        writes = []
        for i, w in enumerate(words):
            if w in (">", ">>", "tee", "-a") and i + 1 < len(words):
                writes.append(words[i + 1])
            m = re.match(r"^\d?>>?(.+)$", w)
            if m:
                writes.append(m.group(1))
        if words[0] in ("mv", "cp", "rm", "ln", "chmod", "touch", "truncate"):
            writes += [w for w in words[1:] if not w.startswith("-")]
        if words[0] == "sed" and any(
            w.startswith("-i") or w == "--in-place" for w in words[1:]
        ):
            writes += [w for w in words[1:] if not w.startswith("-")]
        for t in writes:
            rel = rel_to_project(t, cwd)
            if rel and is_protected(rel):
                decide(
                    "ask",
                    f"'{rel}' is agent or CI configuration. A human must approve this change (AGT-003).",
                )

        # New dependencies (SUP-001).
        joined = " ".join(words)
        for pattern, allow in INSTALL_PATTERNS:
            if re.search(pattern, joined):
                if allow and re.search(allow, joined):
                    break
                decide(
                    "ask",
                    "Adding a dependency. Confirm the package exists in the registry, has real history and is needed. Never guess a name (SUP-001).",
                )

        for pattern, reason in ASK_COMMANDS:
            if re.search(pattern, joined, flags=re.IGNORECASE):
                decide("ask", reason)


def check_file_tool(tool: str, tool_input: dict, cwd: str) -> None:
    path = (
        tool_input.get("file_path")
        or tool_input.get("notebook_path")
        or tool_input.get("path")
        or ""
    )
    if not path:
        return
    if is_secret_file(path):
        if tool == "Read":
            decide(
                "deny",
                f"'{path}' looks like a secret file. Agents never read secrets into the session (SEC-006).",
            )
        decide(
            "deny",
            f"'{path}' looks like a secret file. Secrets come from the secrets manager, never from files written by an agent (SEC-006).",
        )
    if tool == "Read":
        return
    rel = rel_to_project(path, cwd)
    if rel and is_protected(rel):
        decide(
            "ask",
            f"'{rel}' is agent or CI configuration. A human must approve this change (AGT-003).",
        )


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}
    cwd = data.get("cwd") or os.getcwd()

    if tool == "Bash":
        check_bash(tool_input.get("command", ""), cwd)
    elif tool in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        check_file_tool(tool, tool_input, cwd)
    sys.exit(0)


if __name__ == "__main__":
    main()
