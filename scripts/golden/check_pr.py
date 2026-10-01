#!/usr/bin/env python3
"""Golden Rules pull request gates.

Usage:
  python3 scripts/golden/check_pr.py all --base origin/main
  python3 scripts/golden/check_pr.py size|tests|protected --base origin/main
  python3 scripts/golden/check_pr.py unicode [files...]

Override labels (comma separated in GOLDEN_LABELS, set by CI from the PR):
  large-pr-approved       skips the size cap (REV-002)
  test-change-approved    allows removed assertions, skips, snapshot updates (TST-002)
  config-change-approved  allows changes to agent and CI configuration (AGT-003)
Hidden Unicode in instruction files is never overridable (AGT-004).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

MAX_LINES = int(os.environ.get("GOLDEN_MAX_PR_LINES", "400"))
LABELS = {
    x.strip() for x in os.environ.get("GOLDEN_LABELS", "").split(",") if x.strip()
}

LOCKFILES = re.compile(
    r"(^|/)(uv\.lock|poetry\.lock|Pipfile\.lock|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|bun\.lockb?|"
    r"Cargo\.lock|go\.sum|composer\.lock|Gemfile\.lock)$|\.min\.(js|css)$|(^|/)(dist|build|vendor|generated)/|"
    r"^scripts/golden/|^\.claude/hooks/guard\.py$"  # vendored golden-rules tooling
)
TEST_FILE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$|(^|/)__snapshots__/|\.snap$"
)
SNAPSHOT = re.compile(r"(^|/)__snapshots__/|\.snap$|-snapshots/")
ASSERTION = re.compile(
    r"\bassert\b|\bexpect\s*\(|\bassert[A-Z]\w*\s*\(|\.should\b|\bt\.(Error|Fatal|Fail)"
)
SKIP_MARKER = re.compile(
    r"pytest\.mark\.(skip|xfail)|pytest\.skip\(|unittest\.skip|@skip\b|\b(it|test|describe)\.(skip|only|todo)\(|\bx(it|describe|test)\(|t\.Skip\("
)
PROTECTED = (
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
INSTRUCTION_FILE = re.compile(
    r"(^|/)(AGENTS|CLAUDE|GEMINI|SKILL)\.md$|(^|/)\.cursorrules$|(^|/)\.cursor/|(^|/)\.claude/|copilot-instructions\.md$|\.mdc$"
)
HIDDEN = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def changed(base: str) -> list[tuple[str, str]]:
    out = git("diff", "--name-status", "--no-renames", f"{base}...HEAD")
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0][0], parts[-1]))
    return rows


def fail(msg: str, label: str | None = None) -> bool:
    if label and label in LABELS:
        print(f"OVERRIDDEN by label '{label}': {msg}")
        return False
    hint = (
        f" Add the '{label}' label after a human review to override." if label else ""
    )
    print(f"::error::{msg}{hint}")
    return True


def check_size(base: str) -> bool:
    total = 0
    for line in git("diff", "--numstat", f"{base}...HEAD").splitlines():
        added, deleted, path = line.split("\t", 2)
        if added == "-" or LOCKFILES.search(path):
            continue
        total += int(added) + int(deleted)
    print(
        f"size: {total} changed lines (cap {MAX_LINES}, lockfiles and generated files excluded)"
    )
    if total > MAX_LINES:
        return fail(
            f"PR changes {total} lines, above the {MAX_LINES} line cap. Split it (REV-002).",
            "large-pr-approved",
        )
    return False


def check_tests(base: str) -> bool:
    problems = []
    for status, path in changed(base):
        if not TEST_FILE.search(path):
            continue
        if status == "D":
            problems.append(f"deleted test file {path}")
            continue
        if SNAPSHOT.search(path) and status == "M":
            problems.append(f"updated snapshot {path}")
            continue
        diff = git("diff", "-U0", f"{base}...HEAD", "--", path)
        removed = [
            line
            for line in diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
        added = [
            line
            for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed_asserts = sum(1 for line in removed if ASSERTION.search(line))
        added_asserts = sum(1 for line in added if ASSERTION.search(line))
        if removed_asserts > added_asserts:
            problems.append(
                f"{path}: {removed_asserts} assertion lines removed, {added_asserts} added"
            )
        skips = [line[1:].strip() for line in added if SKIP_MARKER.search(line)]
        if skips:
            problems.append(f"{path}: new skip/xfail/only marker: {skips[0][:80]}")
    print(f"tests: {len(problems)} possible weakening(s)")
    if problems:
        return fail(
            "Tests may have been weakened (TST-002): " + "; ".join(problems),
            "test-change-approved",
        )
    return False


def is_protected(path: str) -> bool:
    return any(
        path.startswith(p) if p.endswith("/") else (path == p or path.endswith("/" + p))
        for p in PROTECTED
    )


def check_protected(base: str) -> bool:
    hits = [path for _, path in changed(base) if is_protected(path)]
    print(f"protected: {len(hits)} agent or CI configuration file(s) changed")
    if hits:
        return fail(
            "Agent or CI configuration changed (AGT-003): " + ", ".join(hits),
            "config-change-approved",
        )
    return False


def check_unicode(files: list[str]) -> bool:
    bad = []
    for path in files:
        if not INSTRUCTION_FILE.search(path) or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                m = HIDDEN.search(line)
                if m:
                    bad.append(f"{path}:{n} U+{ord(m.group()):04X}")
    print(f"unicode: {len(bad)} hidden character(s) in instruction files")
    if bad:
        return fail(
            "Hidden or bidirectional Unicode in instruction files (AGT-004): "
            + ", ".join(bad)
        )
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["all", "size", "tests", "protected", "unicode"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--base", default=os.environ.get("GOLDEN_BASE", "origin/main"))
    a = ap.parse_args()

    failed = False
    if a.check in ("all", "size"):
        failed |= check_size(a.base)
    if a.check in ("all", "tests"):
        failed |= check_tests(a.base)
    if a.check in ("all", "protected"):
        failed |= check_protected(a.base)
    if a.check in ("all", "unicode"):
        files = a.files or [p for s, p in changed(a.base) if s != "D"]
        failed |= check_unicode(files)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
