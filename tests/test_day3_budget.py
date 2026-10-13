"""Context window: num_ctx is configured, and oversized prompts are trimmed."""

import logging
from pathlib import Path

import yaml

from jarvis.budget import (
    _cost,
    answer_token_budget,
    assemble_messages,
    context_window,
    estimate_tokens,
    fit_input,
)

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


def _chars(tokens: int) -> str:
    return "x" * (tokens * 4 - 3)


def test_answer_budget_is_at_most_half_the_context_window():
    config = _yaml("jarvis.yaml")
    for _tier, candidates in config["tiers"].items():
        for cand in candidates:
            budget = answer_token_budget(cand["litellm"], config)
            assert budget <= cand["num_ctx"] // 2, cand["litellm"]
    assert answer_token_budget("reasoning-large-mbp", config) == 6000
    assert context_window(config, "reasoning-large-mbp") == 32768
    assert answer_token_budget("reasoning-large-mini", config) == 6000
    assert context_window(config, "reasoning-large-mini") == 16384
    assert answer_token_budget("code-large-mini", config) == 4000
    assert context_window(config, "code-large-mini") == 8192
    assert answer_token_budget("code-small-mini", config) == 4000
    assert context_window(config, "code-small-mini") == 8192


def test_code_large_mini_keeps_history_beside_a_full_rag_block():
    config = _yaml("jarvis.yaml")
    num_ctx = context_window(config, "code-large-mini")
    max_tokens = answer_token_budget("code-large-mini", config)
    rag = _chars(1500)
    history = [
        {"role": "user", "content": _chars(1000)},
        {"role": "assistant", "content": _chars(1000)},
    ]
    assert estimate_tokens(rag) == 1500
    assert sum(estimate_tokens(item["content"]) for item in history) == 2000
    fitted = fit_input(
        system="s",
        history=history,
        rag=rag,
        handoff="",
        user="q",
        max_tokens=max_tokens,
        num_ctx=num_ctx,
    )
    assert fitted.trimmed == ()
    assert [item["content"] for item in fitted.history] == [item["content"] for item in history]


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
