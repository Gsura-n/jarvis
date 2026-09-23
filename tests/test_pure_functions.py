"""
Tests for logic that needs no model, no network and no secrets.
The first eval layer in the design is deterministic checks; these tests are the seed of it.
"""

import json
import os
import time

import pytest

# Tests must never depend on a real ~/.jarvis.env
os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")


def test_settings_load_without_env_file(monkeypatch):
    monkeypatch.setenv("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")
    from jarvis import config

    s = config.load_settings()
    assert s.litellm_base.startswith("http")
    assert s.trace_dir.name == "traces"
    assert "tiers" in s.topology
    assert set(s.missing_required()) >= {"LITELLM_MASTER_KEY", "NEO4J_URI"}


def test_topology_has_every_tier_with_a_mini_fallback():
    from jarvis.config import settings

    tiers = settings.topology["tiers"]
    for name in ["classify", "general-small", "code-small", "reasoning-large", "code-large", "judge"]:
        assert name in tiers, name
        devices = [c["device"] for c in tiers[name]]
        assert "mini" in devices, f"{name} has no candidate on the always-on device"


def test_step_timer_records_steps():
    from jarvis.gateway_metrics import StepTimer

    t = StepTimer()
    t.start("a")
    time.sleep(0.01)
    t.end("a")
    t.end("never-started")  # must be a no-op, not an error
    assert [s["name"] for s in t.steps] == ["a"]
    assert t.steps[0]["duration_s"] >= 0.01
    assert "Performance trace" in t.summary()


decomposer = pytest.importorskip("jarvis.task_decomposer")


@pytest.mark.parametrize(
    "raw",
    [
        '{"complexity": "simple", "reasoning": "x", "estimated_subtasks": 0}',
        '```json\n{"complexity": "simple", "reasoning": "x", "estimated_subtasks": 0}\n```',
        'Sure! Here is the JSON:\n{"complexity": "simple", "reasoning": "x", "estimated_subtasks": 0}\nHope that helps.',
    ],
)
def test_parse_json_tolerates_fences_and_prose(raw):
    assert decomposer.parse_json(raw)["complexity"] == "simple"


def test_parse_json_rejects_garbage():
    with pytest.raises(json.JSONDecodeError):
        decomposer.parse_json("no json here")


@pytest.mark.parametrize(
    "text,broken",
    [
        ("", True),
        ("   ", True),
        ("short", True),
        ("[ERROR] connection refused and a long explanation follows here ok", True),
        ("[No answer extracted — model produced no usable output]", True),
        ("Here is a complete answer that is long enough to count as real output.", False),
    ],
)
def test_is_broken_result(text, broken):
    assert decomposer.is_broken_result(text) is broken


def test_extract_code_blocks_keeps_fences():
    text = "intro\n```python\nprint(1)\n```\nmiddle\n```\nx\n```\n"
    blocks = decomposer.extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].startswith("```python")
