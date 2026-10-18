"""The trace marker, system-text join, TraceStore and trace follow-up detection (pure, no gateway)."""

import pytest

from jarvis.messages import TRACE_MARK, clean_assistant, system_text
from jarvis.traces import TraceStore, asks_for_trace, trace_followup

TRACE = f"{TRACE_MARK}\n- Intent: `general` (classifier)\n- Labels from: `classifier`"


# ── clean_assistant and the marker ────────────────────────────────────────────
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("The answer." + TRACE, "The answer."),
        ("The answer." + TRACE.strip(), "The answer."),
        ("<think>plan</think>\n\nThe answer." + TRACE, "The answer."),
        (TRACE, ""),
        (TRACE.strip(), ""),
        (
            "Use `---` rules, not **Model trace:** headings.",
            "Use `---` rules, not **Model trace:** headings.",
        ),
    ],
)
def test_clean_assistant_cuts_the_trace_appendix(raw, clean):
    assert clean_assistant(raw) == clean


def test_one_constant_marks_the_trace():
    assert TRACE_MARK.strip() == "---\n**Model trace:**"


# ── client system messages ────────────────────────────────────────────────────
def test_system_text_joins_client_system_messages():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": [{"type": "text", "text": "Use metric."}]},
        {"role": "system", "content": None},
        {"role": "assistant", "content": "x"},
    ]
    assert system_text(messages) == "Be brief.\n\nUse metric."


def test_system_text_is_empty_without_system_messages():
    assert system_text([{"role": "user", "content": "hi"}]) == ""


# ── TraceStore ────────────────────────────────────────────────────────────────
def test_trace_store_finds_a_trace_by_the_answer_text():
    store = TraceStore()
    store.put("Hello  there.\n", TRACE)
    assert store.get("Hello there.") == TRACE
    assert store.get("Hello there." + TRACE) == TRACE
    assert store.get("<think>x</think>Hello there.") == TRACE
    assert store.get("Something else") is None


def test_trace_store_drops_the_oldest_past_256():
    store = TraceStore()
    for i in range(257):
        store.put(f"answer {i}", f"trace {i}")
    assert len(store) == 256
    assert store.get("answer 0") is None
    assert store.get("answer 1") == "trace 1"
    assert store.get("answer 256") == "trace 256"


def test_trace_store_keeps_a_recently_read_or_rewritten_entry_longer():
    store = TraceStore(size=2)
    store.put("a", "ta")
    store.put("b", "tb")
    store.put("a", "ta2")
    store.put("c", "tc")
    assert store.get("a") == "ta2" and store.get("b") is None


def test_trace_store_ignores_empty_answers():
    store = TraceStore()
    store.put("<think>only thinking</think>", TRACE)
    store.put(TRACE, TRACE)
    assert len(store) == 0 and store.get("") is None


# ── follow-ups ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text",
    [
        "show trace",
        "Show the trace please",
        "model trace",
        "which model answered?",
        "Which   model\nanswered that",
        "what model answered",
        "which model did you use?",
        "what model did you use",
    ],
)
def test_asks_for_trace(text):
    assert asks_for_trace(text)


@pytest.mark.parametrize(
    "text",
    [
        "which model should I use for embeddings?",
        "why did you pick Postgres?",
        "what model is best for code?",
        "explain how tracing works in OpenTelemetry",
        "",
    ],
)
def test_ordinary_questions_do_not_ask_for_a_trace(text):
    assert not asks_for_trace(text)


def _convo(ask):
    return [
        {"role": "user", "content": "What is a DAG?"},
        {"role": "assistant", "content": "A directed acyclic graph." + TRACE},
        {"role": "user", "content": [{"type": "text", "text": ask}]},
    ]


def test_followup_returns_the_trace_stored_for_the_previous_answer():
    store = TraceStore()
    store.put("A directed acyclic graph.", TRACE)
    assert trace_followup(_convo("which model answered?"), store) == TRACE


def test_followup_without_a_stored_trace_is_none():
    assert trace_followup(_convo("which model answered?"), TraceStore()) is None


def test_followup_needs_the_ask():
    store = TraceStore()
    store.put("A directed acyclic graph.", TRACE)
    assert trace_followup(_convo("which model should I use?"), store) is None


def test_followup_with_no_previous_assistant_turn_is_none():
    store = TraceStore()
    store.put("anything", TRACE)
    assert trace_followup([{"role": "user", "content": "show trace"}], store) is None


def test_followup_looks_at_the_answer_before_the_latest_question_only():
    store = TraceStore()
    store.put("first answer", "trace of first")
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "show trace"},
    ]
    assert trace_followup(messages, store) is None
