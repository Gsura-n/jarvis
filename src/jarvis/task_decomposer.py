"""
task_decomposer.py
Phase 4 — Multi-Agent Task Decomposition using LangGraph

Features:
  - Parallel execution of independent subtasks (ThreadPoolExecutor)
  - Structured handoff: reasoning's full thought chain passed to coding
  - Smart aggregation: code blocks preserved, prose compressed by type,
    progressive (rolling) synthesis instead of one big call
  - Shared context bus across agents
"""

import os
import json
import re
import requests
from typing import TypedDict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from jarvis.config import settings
from jarvis.subtask_tracer import traced_stream_call

# ── Config ────────────────────────────────────────────────────────────────────
LITELLM_BASE = settings.litellm_base
LITELLM_KEY  = settings.litellm_key


# ── State ─────────────────────────────────────────────────────────────────────
class JarvisState(TypedDict):
    prompt:        str
    complexity:    str
    reasoning:     str
    subtasks:      List[dict]
    results:       List[dict]
    rag_context:   str
    final_answer:  str
    error:         Optional[str]


# ── LiteLLM helper ────────────────────────────────────────────────────────────
def call_model(model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 3000, _retry: bool = True) -> str:
    try:
        r = requests.post(
            f"{LITELLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=600,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # Strip reasoning-model think tags so downstream gets clean output
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
        return content.strip()
    except Exception as e:
        if _retry:
            print(f"   [call_model] {model} failed ({e}) — retrying once...")
            import time as _t
            _t.sleep(2)
            return call_model(model, system, user, temperature, max_tokens, _retry=False)
        return f"[ERROR] {e}"


def parse_json(text: str) -> dict:
    clean = text.strip()
    for fence in ["```json", "```"]:
        clean = clean.replace(fence, "")
    clean = clean.strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start >= 0 and end > start:
        clean = clean[start:end]
    return json.loads(clean)


def has_code_block(text: str) -> bool:
    return "```" in text


def extract_code_blocks(text: str) -> list:
    """Pull out fenced code blocks so they're never compressed."""
    return re.findall(r"```[\s\S]*?```", text)


# ── Node 1 — Complexity Detector ──────────────────────────────────────────────
COMPLEXITY_SYSTEM = """You are a task complexity analyzer for an AI coding assistant.

Given a user prompt, determine complexity:

SIMPLE — one clear task, one model call handles it
MEDIUM — 2-3 related tasks, needs some planning
COMPLEX — multiple systems, requires design + build + test

Key: complexity is about SCOPE not prompt length.

Respond ONLY with valid JSON:
{
  "complexity": "simple|medium|complex",
  "reasoning": "brief explanation",
  "estimated_subtasks": 0
}
"""

def detect_complexity(state: JarvisState) -> JarvisState:
    print(f"\n🔍 Detecting complexity for: {state['prompt'][:80]}...")
    try:
        result = call_model("general", COMPLEXITY_SYSTEM, state["prompt"], temperature=0.1)
        parsed = parse_json(result)
        complexity = parsed.get("complexity", "simple")
        reasoning  = parsed.get("reasoning", "")
        print(f"   → Complexity: {complexity} | {reasoning}")
        return {**state, "complexity": complexity, "reasoning": reasoning}
    except Exception as e:
        print(f"   → Detection failed: {e}, defaulting to simple")
        return {**state, "complexity": "simple", "reasoning": "fallback"}


# ── Node 2 — Subtask Splitter ─────────────────────────────────────────────────
SPLITTER_SYSTEM = """You are a task planner for an AI coding assistant.

Break the prompt into clear, ordered subtasks. CRITICAL RULES:

1. Use EXACTLY ONE intent per subtask, NO pipes/combinations:
   - "reasoning" → design decisions, architecture, planning, thinking
   - "code"      → writing actual implementation code
   - "general"   → documentation, explanations, summaries

2. If a task needs thinking AND coding, split into TWO subtasks:
   - First: a "reasoning" subtask to design
   - Second: a "code" subtask with depends_on=[reasoning id] and implements=<reasoning id>

3. Set depends_on correctly — tasks with NO dependencies can run in PARALLEL.
   Independent tasks should have depends_on=[] so they execute simultaneously.

4. Keep subtasks between 3-6 total.

Respond ONLY with valid JSON:
{
  "subtasks": [
    {
      "id": 1,
      "task": "clear description",
      "intent": "reasoning",
      "depends_on": [],
      "implements": null
    },
    {
      "id": 2,
      "task": "another independent design task",
      "intent": "reasoning",
      "depends_on": [],
      "implements": null
    },
    {
      "id": 3,
      "task": "implement based on subtask 1",
      "intent": "code",
      "depends_on": [1],
      "implements": 1
    }
  ]
}

implements MUST be a single number or null, NEVER a list.
"""

def split_tasks(state: JarvisState) -> JarvisState:
    print(f"\n📋 Splitting into subtasks...")
    try:
        result = call_model(
            "general",
            SPLITTER_SYSTEM,
            f"Original prompt: {state['prompt']}\n\nContext: {state['reasoning']}",
            temperature=0.2,
        )
        parsed   = parse_json(result)
        subtasks = parsed.get("subtasks", [])

        # Normalize: ensure implements is single value, intent is clean
        for t in subtasks:
            impl = t.get("implements")
            if isinstance(impl, list):
                t["implements"] = impl[0] if impl else None
            # clean mixed intents
            intent = t.get("intent", "general")
            if "|" in intent:
                # if it mentions code, treat as code; else reasoning
                t["intent"] = "code" if "code" in intent else "reasoning"

        print(f"   → {len(subtasks)} subtasks:")
        for t in subtasks:
            deps = t.get("depends_on", [])
            par = " [PARALLEL]" if not deps else f" [waits for {deps}]"
            impl = f" implements#{t.get('implements')}" if t.get('implements') else ""
            print(f"     [{t['id']}] ({t['intent']}) {t['task'][:50]}...{par}{impl}")

        return {**state, "subtasks": subtasks}
    except Exception as e:
        print(f"   → Splitting failed: {e}")
        return {
            **state,
            "subtasks": [{"id": 1, "task": state["prompt"], "intent": "general", "depends_on": [], "implements": None}]
        }


# ── Node 3 — Parallel Task Executor with Structured Handoff ───────────────────
INTENT_TO_MODEL = {
    "code":      "coding",
    "reasoning": "reasoning",
    "general":   "general",
}

INTENT_SYSTEM = {
    "code":      "You are an expert coder. Write clean, production-ready code with comments.",
    "reasoning": "You are an expert architect. Be DECISIVE. Briefly consider the key options, then COMMIT to a clear recommendation. Do NOT enumerate every possibility or keep second-guessing — pick the best approach and justify it concisely. State your final recommendation explicitly and clearly. Aim to conclude within a few paragraphs.",
    "general":   "You are a helpful expert assistant. Be clear and concise.",
}

HANDOFF_SYSTEM = """You are an expert coder receiving a handoff from a reasoning model.

The reasoning model has already thought through this problem — including approaches
it considered, rejected, and recommended. Your job is to IMPLEMENT the recommended
approach. Do not re-think the architecture; build on the reasoning provided.
Write clean, production-ready code with comments.

=== FULL REASONING HANDOFF ===
{reasoning}
=== END HANDOFF ===

Implement the recommended approach. Avoid the rejected approaches."""


def is_broken_result(text: str) -> bool:
    """Detect a failed/empty/spiral subtask result so we don't poison downstream."""
    if not text or len(text.strip()) < 40:
        return True
    markers = ["[ERROR]", "[No answer", "[Note: model did not"]
    return any(text.lstrip().startswith(m) for m in markers)


def run_single_subtask(subtask: dict, completed: dict, rag_context: str) -> dict:
    """Execute one subtask. Used by both parallel and sequential paths."""
    intent     = subtask.get("intent", "general")
    model      = INTENT_TO_MODEL.get(intent, "general")
    implements = subtask.get("implements")
    if isinstance(implements, list):
        implements = implements[0] if implements else None

    # Build context from dependencies — but SKIP broken ones to prevent
    # cascade poisoning (one failed subtask corrupting everything downstream).
    dep_context = ""
    for dep_id in subtask.get("depends_on", []):
        if dep_id in completed:
            dep_result = completed[dep_id]["result"]
            if is_broken_result(dep_result):
                print(f"      ⚠️  skipping broken context from subtask [{dep_id}]")
                continue
            dep_context += f"\n### Result from subtask [{dep_id}]: {completed[dep_id]['task']}\n{dep_result[:1000]}\n"

    # Structured handoff — only if the reasoning subtask produced a GOOD result.
    # If the handoff source is broken, fall back to standalone execution using
    # the original task + RAG, rather than building on garbage.
    handoff_ok = (
        implements
        and implements in completed
        and not is_broken_result(completed[implements]["result"])
    )

    if handoff_ok:
        reasoning_full = completed[implements]["result"]
        system = HANDOFF_SYSTEM.format(reasoning=reasoning_full)
        if rag_context:
            system += f"\n\nCodebase context:\n{rag_context}"
        user_prompt = f"Implement: {subtask['task']}"
        tag = f"(handoff from #{implements})"
    else:
        system = INTENT_SYSTEM.get(intent, INTENT_SYSTEM["general"])
        if rag_context:
            system += f"\n\nCodebase context:\n{rag_context}"
        user_prompt = f"{subtask['task']}{dep_context}"
        if implements and not handoff_ok:
            tag = "(standalone — handoff source was broken)"
        else:
            tag = ""

    print(f"   → [{subtask['id']}] {subtask['task'][:50]}... ({model}) {tag}")
    # Reasoning models spend tokens thinking before answering, so give them
    # a larger budget AND more wall-clock time. Code/general are faster.
    if model == "reasoning":
        token_budget = 6000
        wall_limit   = 420   # 7 min — DeepSeek is thorough but slow
    else:
        token_budget = 4000
        wall_limit   = 300   # 5 min
    result_text = traced_stream_call(
        model, system, user_prompt,
        subtask_id=subtask["id"],
        task_desc=subtask["task"][:80],
        max_tokens=token_budget,
        max_wall_seconds=wall_limit,
    )
    print(f"     ✅ [{subtask['id']}] done ({len(result_text)} chars)")

    return {
        "id":         subtask["id"],
        "task":       subtask["task"],
        "intent":     intent,
        "implements": implements,
        "result":     result_text,
    }


def execute_tasks(state: JarvisState) -> JarvisState:
    """
    Execute subtasks in dependency order — SEQUENTIAL.
    On a single 16GB machine, parallel execution causes model swapping
    and timeouts. Parallelism is enabled in Phase 6 (multi-device) when
    multiple Ollama endpoints can truly run simultaneously.
    """
    print(f"\n⚙️  Executing {len(state['subtasks'])} subtask(s) sequentially...")

    subtasks    = state["subtasks"]
    completed   = {}
    results     = []
    rag_context = state.get("rag_context", "")
    remaining   = list(subtasks)
    max_rounds  = len(subtasks) + 2
    rnd         = 0

    while remaining and rnd < max_rounds:
        rnd += 1
        ready = [
            t for t in remaining
            if all(d in completed for d in t.get("depends_on", []))
        ]
        if not ready:
            break

        # Run one at a time — no model swapping
        for subtask in ready:
            res = run_single_subtask(subtask, completed, rag_context)
            completed[res["id"]] = res
            results.append(res)
            remaining.remove(subtask)

    results.sort(key=lambda r: r["id"])
    return {**state, "results": results}


def execute_tasks_parallel(state: JarvisState) -> JarvisState:
    """PARALLEL version — kept for Phase 6 multi-device. Do not use on single machine."""
    print(f"\n⚙️  Executing {len(state['subtasks'])} subtask(s) with parallelism...")

    subtasks    = state["subtasks"]
    completed   = {}
    results     = []
    rag_context = state.get("rag_context", "")
    remaining   = list(subtasks)
    max_rounds  = len(subtasks) + 2
    rnd         = 0

    while remaining and rnd < max_rounds:
        rnd += 1
        ready = [t for t in remaining if all(d in completed for d in t.get("depends_on", []))]
        if not ready:
            break
        print(f"\n   Round {rnd}: running {len(ready)} subtask(s) in parallel")
        if len(ready) == 1:
            res = run_single_subtask(ready[0], completed, rag_context)
            completed[res["id"]] = res
            results.append(res)
            remaining.remove(ready[0])
        else:
            with ThreadPoolExecutor(max_workers=min(len(ready), 3)) as ex:
                futures = {ex.submit(run_single_subtask, t, completed, rag_context): t for t in ready}
                for fut in as_completed(futures):
                    res = fut.result()
                    completed[res["id"]] = res
                    results.append(res)
                    remaining.remove(futures[fut])
    results.sort(key=lambda r: r["id"])
    return {**state, "results": results}


# ── Node 4 — Smart Progressive Aggregator ─────────────────────────────────────
PROSE_COMPRESS_SYSTEM = """Summarize this text concisely but keep ALL important details,
decisions, and technical specifics. Do not over-compress. Aim for 40-60% of original length."""

SYNTHESIS_SYSTEM = """You are a technical writer building a coherent final answer.

You will receive the running synthesis so far, plus a new subtask result.
Integrate the new result into the synthesis naturally. Keep ALL code blocks intact
and complete. Maintain logical flow. Don't repeat what's already covered."""


def smart_compress(result: dict) -> str:
    """Compress prose but preserve code blocks fully."""
    text = result["result"]

    # Code subtasks — keep code blocks fully, compress only prose around them
    if result["intent"] == "code" and has_code_block(text):
        code_blocks = extract_code_blocks(text)
        # Replace code blocks with placeholders, compress prose, restore
        prose = text
        for i, block in enumerate(code_blocks):
            prose = prose.replace(block, f"__CODE_BLOCK_{i}__")
        if len(prose) > 600:
            prose = call_model("general", PROSE_COMPRESS_SYSTEM, prose, temperature=0.1)
        # restore code blocks
        for i, block in enumerate(code_blocks):
            prose = prose.replace(f"__CODE_BLOCK_{i}__", f"\n{block}\n")
        return prose

    # Reasoning — keep more (decisions matter), compress lightly
    if result["intent"] == "reasoning":
        if len(text) < 800:
            return text
        return call_model("general", PROSE_COMPRESS_SYSTEM, text, temperature=0.1)

    # General — compress more aggressively
    if len(text) < 400:
        return text
    return call_model("general", PROSE_COMPRESS_SYSTEM, text, temperature=0.1)


def aggregate_results(state: JarvisState) -> JarvisState:
    print(f"\n🔗 Aggregating {len(state['results'])} results...")

    # Drop broken/empty subtask results so they don't pollute the final answer
    all_results = state["results"]
    results = [r for r in all_results if not is_broken_result(r["result"])]
    dropped = len(all_results) - len(results)
    if dropped:
        print(f"   ⚠️  dropped {dropped} broken subtask result(s) from aggregation")

    if not results:
        return {**state, "final_answer": "[All subtasks failed to produce usable output. Check the traces directory.]"}

    if len(results) == 1:
        return {**state, "final_answer": results[0]["result"]}

    # For 2-3 subtasks: just concatenate with clean headers — no LLM calls needed.
    # The individual results are already high quality; synthesis adds cost + timeout risk.
    if len(results) <= 3:
        print("   Small job — concatenating with headers (no extra LLM calls)")
        final = f"# {state['prompt']}\n\n"
        for r in results:
            final += f"## {r['task']}\n\n{r['result']}\n\n---\n\n"
        return {**state, "final_answer": final}

    # For 4+ subtasks: smart compress (only large results) then progressive synthesis
    print("   Large job — smart compress + progressive synthesis...")
    compressed = []
    for r in results:
        # Skip compression for already-small results — saves LLM calls
        if len(r["result"]) < 1500:
            compressed.append({"task": r["task"], "intent": r["intent"], "content": r["result"]})
            print(f"   [{r['id']}] kept as-is ({len(r['result'])} chars)")
        else:
            c = smart_compress(r)
            compressed.append({"task": r["task"], "intent": r["intent"], "content": c})
            print(f"   [{r['id']}] {len(r['result'])} → {len(c)} chars ({r['intent']})")

    # Progressive synthesis
    synthesis = f"# {state['prompt']}\n\n"
    synthesis += f"## {compressed[0]['task']}\n\n{compressed[0]['content']}\n"

    for c in compressed[1:]:
        synthesis_input = (
            f"Running synthesis so far:\n{synthesis}\n\n"
            f"New subtask result to integrate:\n## {c['task']}\n{c['content']}"
        )
        synthesis = call_model("general", SYNTHESIS_SYSTEM, synthesis_input, temperature=0.2)

    print(f"   ✅ Final answer ({len(synthesis)} chars)")
    return {**state, "final_answer": synthesis}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_by_complexity(state: JarvisState) -> str:
    return "execute" if state.get("complexity") == "simple" else "split"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(JarvisState)
    graph.add_node("detect",    detect_complexity)
    graph.add_node("split",     split_tasks)
    graph.add_node("execute",   execute_tasks)
    graph.add_node("aggregate", aggregate_results)
    graph.set_entry_point("detect")
    graph.add_conditional_edges("detect", route_by_complexity, {"split": "split", "execute": "execute"})
    graph.add_edge("split",     "execute")
    graph.add_edge("execute",   "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public interface ──────────────────────────────────────────────────────────
def decompose_and_execute(prompt: str, rag_context: str = "") -> dict:
    graph = get_graph()
    initial_state: JarvisState = {
        "prompt":       prompt,
        "complexity":   "",
        "reasoning":    "",
        "subtasks":     [],
        "results":      [],
        "rag_context":  rag_context,
        "final_answer": "",
        "error":        None,
    }
    final_state = graph.invoke(initial_state)
    models_used = list(set(
        INTENT_TO_MODEL.get(r["intent"], "general")
        for r in final_state.get("results", [])
    ))
    return {
        "answer":      final_state.get("final_answer", ""),
        "complexity":  final_state.get("complexity", "simple"),
        "subtasks":    final_state.get("subtasks", []),
        "results":     final_state.get("results", []),
        "models_used": models_used,
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧠 Testing Phase 4 Multi-Agent Decomposer\n")
    print("=" * 50)
    result = decompose_and_execute(
        "Design and build a REST API for a todo app with authentication and CRUD operations"
    )
    print(f"\nComplexity: {result['complexity']}")
    print(f"Subtasks: {len(result['subtasks'])}")
    print(f"\nFinal answer preview:\n{result['answer'][:600]}...")
