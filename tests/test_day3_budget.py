"""Context window: num_ctx is configured, and oversized prompts are trimmed."""

import logging
from pathlib import Path

import yaml

from jarvis.budget import _cost, assemble_messages, estimate_tokens, fit_input

ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def test_every_litellm_deployment_has_num_ctx():
    for item in _yaml("configs/litellm.yaml")["model_list"]:
        num_ctx = item["litellm_params"].get("num_ctx")
        assert isinstance(num_ctx, int) and num_ctx >= 8192, item["model_name"]


def test_every_jarvis_deployment_has_num_ctx_and_a_litellm_entry():
    lite = {item["model_name"]: item for item in _yaml("configs/litellm.yaml")["model_list"]}
    for tier, candidates in _yaml("jarvis.yaml")["tiers"].items():
        for cand in candidates:
            assert isinstance(cand.get("num_ctx"), int) and cand["num_ctx"] >= 8192, tier
            assert cand["litellm"] in lite, cand["litellm"]
            assert lite[cand["litellm"]]["litellm_params"]["num_ctx"] == cand["num_ctx"]


def test_oldest_history_is_dropped_before_newer_history():
    fitted = fit_input(
        system="sys",
        history=[
            {"role": "user", "content": "OLD" * 200},
            {"role": "assistant", "content": "NEW"},
        ],
        rag="R" * 80,
        handoff="",
        user="question",
        max_tokens=20,
        num_ctx=80,
    )
    assert [item["content"] for item in fitted.history] == ["NEW"]
    assert fitted.trimmed == ("history",)
    assert _cost(fitted.system, list(fitted.history), fitted.rag, fitted.handoff, fitted.user) <= 60


def test_oversized_input_is_trimmed_and_logged(caplog):
    caplog.set_level(logging.WARNING)
    messages = assemble_messages(
        system="You are a helper.",
        history=[{"role": "user", "content": "old " * 300}, {"role": "assistant", "content": "newer"}],
        rag="context " * 400,
        handoff="design " * 400,
        user="do the thing",
        max_tokens=40,
        num_ctx=160,
    )
    assert "trimmed input" in caplog.text
    assert messages[-1]["content"] == "do the thing"
    assert "You are a helper." in messages[0]["content"]
    assert estimate_tokens(messages[0]["content"]) + estimate_tokens(messages[-1]["content"]) <= 160
