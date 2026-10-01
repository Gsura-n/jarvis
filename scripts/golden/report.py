#!/usr/bin/env python3
"""Turn scanner JSON into GitHub annotations and a pass/fail exit code.

  python3 scripts/golden/report.py semgrep semgrep.json
  python3 scripts/golden/report.py pip-audit audit.json

Each finding is printed as a ::error annotation so it shows on the PR and in
the checks API, not only in the raw job log.
"""

from __future__ import annotations

import json
import sys


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def semgrep(path: str) -> int:
    data = json.load(open(path))
    results = data.get("results", [])
    for r in results:
        msg = r.get("extra", {}).get("message", "").strip()
        rule = r.get("check_id", "semgrep")
        line = r.get("start", {}).get("line", 1)
        print(f"::error file={r.get('path')},line={line},title={rule}::{esc(msg)}")
    errors = data.get("errors", [])
    for e in errors:
        if e.get("level") == "error":
            print(
                f"::warning title=semgrep error::{esc(str(e.get('message', e))[:500])}"
            )
    print(f"semgrep: {len(results)} new finding(s)")
    return 1 if results else 0


def pip_audit(path: str) -> int:
    data = json.load(open(path))
    deps = data.get("dependencies", data if isinstance(data, list) else [])
    count = 0
    for d in deps:
        for v in d.get("vulns", []):
            count += 1
            fixes = ", ".join(v.get("fix_versions", [])) or "no fix yet"
            aliases = ", ".join(v.get("aliases", []))
            print(
                f"::error title={d['name']} {d['version']}::{v['id']} ({aliases}) "
                f"fixed in {fixes}. {esc(v.get('description', '')[:300])}"
            )
    print(f"pip-audit: {count} known vulnerabilit{'y' if count == 1 else 'ies'}")
    return 1 if count else 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("semgrep", "pip-audit"):
        print(__doc__)
        return 2
    return semgrep(sys.argv[2]) if sys.argv[1] == "semgrep" else pip_audit(sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
