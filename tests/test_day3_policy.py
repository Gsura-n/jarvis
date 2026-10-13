"""Routing policy: tier, then the first candidate whose device is up and has the model."""

import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

from jarvis.config import settings  # noqa: E402
from jarvis.policy import decide, prefer_decomposer  # noqa: E402

MBP_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
MINI_MODELS = (
    "llama3.2:3b",
    "nomic-embed-text",
    "llama3.1:8b",
    "qwen2.5-coder:7b",
    "deepseek-r1:8b",
)


def snap(online: set[str], pulled: dict[str, tuple]) -> dict:
    return {
        "devices": {
            name: {"online": name in online, "pulled": pulled.get(name, ())}
            for name in ("mini", "macbook-pro")
        }
    }


BOTH = snap({"mini", "macbook-pro"}, {"mini": MINI_MODELS, "macbook-pro": (MBP_MODEL,)})
MINI = snap({"mini"}, {"mini": MINI_MODELS, "macbook-pro": (MBP_MODEL,)})


def test_laptop_online_serves_reasoning_large():
    decision = decide("reasoning", "", None, BOTH, settings.topology)
    assert decision.tier == "reasoning-large"
    assert decision.device == "macbook-pro"
    assert decision.deployment == "reasoning-large-mbp"
    assert decision.degraded is False
    design = decide("code+reasoning", "large", None, BOTH, settings.topology)
    assert design.deployment == "reasoning-large-mbp"
    assert prefer_decomposer("code+reasoning", None, design, force=False) is False


def test_laptop_offline_falls_back_to_mini():
    decision = decide("reasoning", "large", None, MINI, settings.topology)
    assert decision.device == "mini"
    assert decision.deployment == "reasoning-large-mini"
    assert decision.degraded is True
    assert prefer_decomposer("code+reasoning", None, decision, force=False) is True


def test_model_not_pulled_skips_the_laptop():
    missing = snap({"mini", "macbook-pro"}, {"mini": MINI_MODELS, "macbook-pro": ()})
    decision = decide("code+reasoning", "", None, missing, settings.topology)
    assert decision.tier == "reasoning-large"
    assert decision.device == "mini"
    assert decision.degraded is True
    assert "not pulled" in decision.reason


def test_explicit_model_beats_intent():
    decision = decide("reasoning", "large", "coding", BOTH, settings.topology)
    assert decision.tier == "code-small"
    assert decision.deployment == "code-small-mini"
    assert decision.device == "mini"
    assert decision.degraded is False
    assert prefer_decomposer("code", "coding", decision, force=False) is False
    forced = decide("general", "", None, BOTH, settings.topology)
    assert prefer_decomposer("general", None, forced, force=True) is True


def test_nothing_available_still_names_the_mini_fallback():
    decision = decide("general", "", None, {"devices": {}}, settings.topology)
    assert decision.degraded is True
    assert decision.device == "mini"
    assert decision.deployment == "general-small-mini"
    assert decision.reason
