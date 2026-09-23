"""
subtask_tracer.py
Lightweight observer for the Jarvis task decomposer.

Streams model output to disk as it arrives — captures the thinking process
(DeepSeek <think> tags), the final answer, and diagnostic stats — WITHOUT
holding the full thinking in memory.

Memory-safe: thinking tokens are written to a file as they stream, never
accumulated in a variable. A model that thinks for 10,000 tokens uses the
same RAM as one that thinks for 100.

Usage in decomposer:
    from subtask_tracer import traced_stream_call
    result = traced_stream_call(model, system, user, subtask_id, task_desc)
"""

import json
import os
import re
import time
from datetime import datetime

import requests

from jarvis.config import settings

LITELLM_BASE = settings.litellm_base
LITELLM_KEY = settings.litellm_key

# Trace directory — created on demand (default: <repo>/.logs/traces, set in jarvis.yaml)
TRACE_DIR = str(settings.trace_dir)

# Master toggle — if False, tracing is skipped entirely (zero overhead)
TRACING_ENABLED = True


def _ensure_trace_dir():
    os.makedirs(TRACE_DIR, exist_ok=True)


def traced_stream_call(
    model: str,
    system: str,
    user: str,
    subtask_id: int = 0,
    task_desc: str = "",
    temperature: float = 0.3,
    timeout: int = 600,
    max_tokens: int = 3000,
    max_wall_seconds: int = 300,
) -> str:
    """
    Stream a model call through LiteLLM. Write thinking + answer to a trace
    file as tokens arrive. Return ONLY the final answer (thinking stripped).

    Memory-safe: only the answer is kept in memory. Thinking streams to disk.
    """
    if not TRACING_ENABLED:
        # Fallback to plain non-streaming call
        return _plain_call(model, system, user, temperature, timeout)

    _ensure_trace_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    trace_path = os.path.join(TRACE_DIR, f"{ts}_subtask{subtask_id}_{model}.log")

    start = time.time()
    # Counters — small integers, not accumulating text
    think_chars = 0
    answer_chars = 0
    in_think = False

    # Keep a BOUNDED tail of recent thinking (last ~4000 chars) in memory.
    # Used as a fallback if the model spirals and never produces an answer.
    # Bounded so memory stays flat even on a 14000-char think spiral.
    from collections import deque

    think_tail = deque(maxlen=4000)  # chars, not tokens

    # We keep ONLY the answer in memory (needed for return value).
    # Thinking is written straight to disk and discarded.
    answer_parts = []

    try:
        with open(trace_path, "w", encoding="utf-8") as tf:
            # Header
            tf.write(f"=== SUBTASK {subtask_id} ===\n")
            tf.write(f"Model: {model}\n")
            tf.write(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
            tf.write(f"Task: {task_desc}\n")
            tf.write(f"{'=' * 50}\n\n")
            tf.flush()

            r = requests.post(
                f"{LITELLM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LITELLM_KEY}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=timeout,
                stream=True,
            )
            r.raise_for_status()

            for line in r.iter_lines():
                # Wall-clock guard — abort if a single subtask runs too long
                if time.time() - start > max_wall_seconds:
                    tf.write(f"\n\n--- ABORTED: exceeded {max_wall_seconds}s wall limit ---\n")
                    tf.flush()
                    r.close()
                    break
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                if decoded == "data: [DONE]":
                    break
                try:
                    chunk = json.loads(decoded[6:])
                    delta = chunk["choices"][0]["delta"]
                except Exception:
                    continue

                # LiteLLM separates reasoning models into two fields:
                #   reasoning_content = the thinking (goes to disk)
                #   content           = the actual answer (kept in memory)
                think_tok = delta.get("reasoning_content", "")
                answer_tok = delta.get("content", "")

                if think_tok:
                    if not in_think:
                        in_think = True
                        tf.write("\n--- THINKING ---\n")
                    tf.write(think_tok)
                    think_chars += len(think_tok)
                    think_tail.extend(think_tok)  # bounded memory
                    if think_chars % 200 < len(think_tok):
                        tf.flush()

                if answer_tok:
                    if in_think:
                        in_think = False
                        tf.write("\n--- ANSWER ---\n")
                    answer_parts.append(answer_tok)
                    tf.write(answer_tok)
                    answer_chars += len(answer_tok)
                    tf.flush()

            # Stats footer
            duration = round(time.time() - start, 1)
            ratio = round(think_chars / max(answer_chars, 1), 2)
            tf.write("\n\n--- STATS ---\n")
            tf.write(f"Duration: {duration}s\n")
            tf.write(f"Thinking chars: {think_chars}\n")
            tf.write(f"Answer chars: {answer_chars}\n")
            tf.write(f"Think/answer ratio: {ratio}\n")
            if ratio > 4:
                tf.write("⚠️  HIGH RATIO — model may be over-thinking/spiraling\n")
            tf.flush()

        # reasoning_content and content are already separated by LiteLLM,
        # so answer_parts contains ONLY the real answer — no stripping needed.
        answer = "".join(answer_parts).strip()

        if not answer:
            # Model spiraled or was cut off before producing an answer.
            # Salvage the tail of its thinking rather than returning nothing —
            # the reasoning usually contains a usable conclusion near the end.
            salvaged = "".join(think_tail).strip()
            if salvaged:
                answer = (
                    "[Note: model did not produce a final answer within limits; "
                    "using the conclusion of its reasoning]\n\n" + salvaged
                )
            else:
                answer = "[No answer extracted — model produced no usable output]"

        print(f"     📝 trace: {trace_path} (think/answer ratio: {ratio})")
        return answer

    except Exception as e:
        # Log the failure too
        try:
            with open(trace_path, "a", encoding="utf-8") as tf:
                tf.write(f"\n\n--- ERROR ---\n{e}\nDuration before fail: {round(time.time() - start, 1)}s\n")
        except Exception:
            pass
        return f"[ERROR] {e}"


def _plain_call(model, system, user, temperature, timeout):
    """Non-streaming fallback when tracing is disabled."""
    try:
        r = requests.post(
            f"{LITELLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def get_latest_traces(n: int = 5) -> list:
    """Return paths to the N most recent trace files for analysis."""
    if not os.path.isdir(TRACE_DIR):
        return []
    files = [os.path.join(TRACE_DIR, f) for f in os.listdir(TRACE_DIR) if f.endswith(".log")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[:n]


def summarize_traces(n: int = 5) -> str:
    """Quick summary of recent traces — durations and ratios for diagnosis."""
    traces = get_latest_traces(n)
    if not traces:
        return "No traces found."
    lines = ["Recent subtask traces:\n"]
    for path in traces:
        try:
            with open(path) as f:
                content = f.read()
            dur = re.search(r"Duration: ([\d.]+)s", content)
            ratio = re.search(r"Think/answer ratio: ([\d.]+)", content)
            spiral = "⚠️ SPIRAL" if "HIGH RATIO" in content else ""
            name = os.path.basename(path)
            lines.append(f"  {name}")
            lines.append(
                f"    duration={dur.group(1) if dur else '?'}s  ratio={ratio.group(1) if ratio else '?'}  {spiral}"
            )
        except Exception:
            pass
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing traced stream call...")
    result = traced_stream_call(
        "reasoning",
        "You are an architect. Think step by step.",
        "Design a simple URL shortener. Be concise.",
        subtask_id=99,
        task_desc="test task",
    )
    print(f"\nAnswer ({len(result)} chars):\n{result[:300]}")
    print("\n" + summarize_traces(3))
