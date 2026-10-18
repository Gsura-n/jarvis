"""Routing policy. No I/O. First online candidate with the model pulled wins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jarvis.registry import normalize_model


@dataclass(frozen=True)
class Decision:
    tier: str
    deployment: str
    device: str
    degraded: bool
    reason: str


# Client model names from /v1/models. These skip intent classification.
_EXPLICIT = {
    "general": "general-small",
    "coding": "code-small",
    "reasoning": "reasoning-large",
}
# Only "complex" goes to a large tier. "medium" stays small on purpose; unknown values do too.
_LARGE = {"complex"}


def pick_tier(intent: str, complexity: str, explicit_model: str | None) -> tuple[str, str]:
    """Return (tier, why). explicit_model wins over intent."""
    if explicit_model in _EXPLICIT:
        return _EXPLICIT[explicit_model], f"explicit model {explicit_model}"
    kind = (intent or "general").strip().lower()
    level = (complexity or "").strip().lower()
    large = level in _LARGE
    if kind == "code+reasoning":
        # One 30B call covers design and build when that tier is really available.
        return "reasoning-large", "intent code+reasoning"
    if kind == "code":
        tier = "code-large" if large else "code-small"
        return tier, f"intent code complexity {level or 'simple'}"
    if kind == "reasoning":
        return "reasoning-large", "intent reasoning"
    return "general-small", f"intent {kind or 'general'}"


def _view(snapshot, device: str):
    devices = (
        snapshot.get("devices", {}) if isinstance(snapshot, Mapping) else getattr(snapshot, "devices", {})
    )
    return devices.get(device)


def _ok(snapshot, device: str, model: str) -> tuple[bool, str]:
    view = _view(snapshot, device)
    if view is None:
        return False, f"{device} offline"
    online = view["online"] if isinstance(view, Mapping) else view.online
    pulled = view["pulled"] if isinstance(view, Mapping) else view.pulled
    if not online:
        return False, f"{device} offline"
    wanted = normalize_model(model)
    if wanted not in tuple(pulled or ()):
        return False, f"{model} not pulled on {device}"
    return True, ""


def walk_tier(tier: str, snapshot, config: Mapping, why: str) -> Decision:
    """Walk one tier's candidate list. Used by decide() and by the classifier."""
    candidates = list(config.get("tiers", {}).get(tier, []))
    skipped: list[str] = []
    for cand in candidates:
        ok, skip = _ok(snapshot, cand["device"], cand["model"])
        if not ok:
            skipped.append(skip)
            continue
        note = ("; ".join(skipped) + "; ") if skipped else ""
        return Decision(
            tier,
            cand["litellm"],
            cand["device"],
            bool(cand.get("degraded")),
            f"{why}; {note}selected {cand['litellm']} on {cand['device']}",
        )
    minis = [c for c in candidates if c.get("device") == "mini"]
    fallback = ([c for c in minis if c.get("degraded")] or minis or candidates or [None])[-1]
    if not fallback:
        return Decision(tier, "", "", True, f"{why}; no candidates configured")
    detail = "; ".join(skipped) or "no candidate available"
    return Decision(
        tier,
        fallback["litellm"],
        fallback["device"],
        True,
        f"{why}; {detail}; fallback {fallback['litellm']} on {fallback['device']}",
    )


def decide(intent: str, complexity: str, explicit_model: str | None, snapshot, config: Mapping) -> Decision:
    tier, why = pick_tier(intent, complexity, explicit_model)
    return walk_tier(tier, snapshot, config, why)


def prefer_decomposer(intent: str, explicit_model: str | None, decision: Decision, force: bool) -> bool:
    """One 30B call for design-and-build, unless that call is degraded or the client forced a split."""
    return bool(force or (explicit_model is None and intent == "code+reasoning" and decision.degraded))
