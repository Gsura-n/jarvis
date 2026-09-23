"""
Smart Gateway — port 4001
OpenAI-compatible API that orchestrates:
  1. Prompt enhancement
  2. Intent classification
  3. RAG context (project-aware)
  4. Model routing + chaining
  5. Streaming responses
  6. Model trace on demand with timing + memory
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from neo4j import GraphDatabase

from jarvis.config import settings
from jarvis.gateway_metrics import StepTimer
from jarvis.task_decomposer import decompose_and_execute

# ── Config — everything comes from jarvis.config (env file + jarvis.yaml) ─────
LITELLM_BASE = settings.litellm_base
LITELLM_KEY = settings.litellm_key
NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password
OLLAMA_BASE = settings.ollama_base
EMBED_MODEL = settings.embed_model

# Validate required credentials at startup
_missing = settings.missing_required()
if _missing:
    print(f"⚠️  Missing required env vars: {', '.join(_missing)}")
    print(f"   Add them to {settings.env_file}")

MODELS = {
    "general": "general",
    "code": "coding",
    "reasoning": "reasoning",
}

TRACE_TRIGGERS = [
    "which model",
    "what model",
    "why did you",
    "how did you",
    "explain your process",
    "show trace",
    "model trace",
    "who answered",
]

RAG_TRIGGERS = [
    "my project",
    "my code",
    "my repo",
    "my codebase",
    "my file",
    "this project",
    "this repo",
    "this code",
    "this file",
    "in the project",
    "in the codebase",
    "in my app",
    "portfolio",
    "agent page",
    "dashboard",
    "component",
]

app = FastAPI(title="Jarvis Smart Gateway")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ── Neo4j ─────────────────────────────────────────────────────────────────────
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
except Exception:
    driver = None


def get_embedding(text: str) -> list:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text[:2000]},
            timeout=15,
        )
        return r.json().get("embeddings", [[]])[0]
    except Exception:
        return []


def rag_query(question: str, top_k: int = 4) -> str:
    if not driver:
        return ""
    try:
        embedding = get_embedding(question)
        if not embedding:
            return ""
        with driver.session() as session:
            result = session.run(
                """
                MATCH (f:File)
                WHERE f.embedding IS NOT NULL
                WITH f, vector.similarity.cosine(f.embedding, $emb) AS score
                ORDER BY score DESC LIMIT $k
                RETURN f.path AS path, f.content_preview AS preview, score
                """,
                emb=embedding,
                k=top_k,
            )
            rows = [dict(r) for r in result]
        if not rows:
            return ""
        parts = ["### Relevant codebase context (via RAG):"]
        for r in rows:
            parts.append(f"\n**File:** `{r['path']}`\n```\n{r['preview']}\n```")
        return "\n".join(parts)
    except Exception as e:
        print(f"[RAG] error: {e}")
        return ""


# ── LiteLLM helpers ───────────────────────────────────────────────────────────
def chat(model_alias: str, messages: list, temperature: float = 0.3, _retry: bool = True) -> str:
    """Non-streaming chat — returns full response string. Retries once on failure."""
    try:
        r = requests.post(
            f"{LITELLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            json={"model": model_alias, "messages": messages, "temperature": temperature},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if _retry:
            print(f"[chat] {model_alias} failed ({e}) — retrying once...")
            time.sleep(2)
            return chat(model_alias, messages, temperature, _retry=False)
        raise


def chat_stream(model_alias: str, messages: list, temperature: float = 0.3):
    """Streaming chat — yields tokens as they arrive from LiteLLM."""
    r = requests.post(
        f"{LITELLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_KEY}"},
        json={"model": model_alias, "messages": messages, "temperature": temperature, "stream": True},
        timeout=600,
        stream=True,
    )
    for line in r.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                try:
                    chunk = json.loads(decoded[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    pass


# ── Step 1 — Enhance + Classify ───────────────────────────────────────────────
ENHANCE_SYSTEM = """You are a prompt engineer and intent classifier.

Given a user message (possibly casual or brief), do two things:
1. Rewrite it as a clear, detailed prompt that preserves the original intent but adds helpful context and specificity. Keep it concise — do not pad.
2. Classify the intent as ONE of: code | reasoning | code+reasoning | general

Rules for classification:
- code: writing, fixing, reviewing, or explaining specific code — use when asked to show, list, find, or describe code/files
- reasoning: design decisions, architecture, debugging logic, "why", "how should I" — pure thinking, no code needed
- code+reasoning: ONLY when user explicitly wants something DESIGNED AND BUILT from scratch — e.g. "design and implement", "create a new feature", "build this"
- general: concepts, summaries, chat, questions that don't need code or deep reasoning

IMPORTANT: "what does X do", "list my routes", "explain this file", "what are my components" = code (NOT code+reasoning)
ONLY use code+reasoning when the user clearly wants NEW code written after thinking through a solution.

needs_rag should be true when ANY of these apply:
- Query mentions "my project", "my code", "my repo", "my file", "this project", "this code"
- Query is about a specific file, component, function, or feature in a codebase
- Query mentions specific framework terms like "agent page", "dashboard", "component", "route", "API"
- Intent is "code" or "code+reasoning"

Respond ONLY with valid JSON, no markdown:
{
  "enhanced_prompt": "...",
  "intent": "code|reasoning|code+reasoning|general",
  "needs_rag": true|false,
  "rag_reason": "brief reason or empty string"
}
"""


def enhance_and_classify(user_message: str) -> dict:
    msg_lower = user_message.lower()
    keyword_rag = any(t in msg_lower for t in RAG_TRIGGERS)

    try:
        result = chat(
            "classifier",
            [
                {"role": "system", "content": ENHANCE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )

        clean = result.strip()
        for fence in ["```json", "```"]:
            clean = clean.replace(fence, "")
        clean = clean.strip()

        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]

        parsed = json.loads(clean)

    except Exception as e:
        print(f"[Classify] parse error: {e} — using fallback")
        intent = "general"
        if any(w in msg_lower for w in ["write", "build", "create", "implement", "fix", "code", "function"]):
            intent = "code"
        elif any(w in msg_lower for w in ["why", "how should", "design", "architect", "explain"]):
            intent = "reasoning"
        parsed = {
            "enhanced_prompt": user_message,
            "intent": intent,
            "needs_rag": keyword_rag,
            "rag_reason": "fallback classifier",
        }

    if keyword_rag:
        parsed["needs_rag"] = True
        if not parsed.get("rag_reason"):
            parsed["rag_reason"] = "keyword match"

    return parsed


# ── Step 2 — Execute (non-streaming) ─────────────────────────────────────────
def execute(intent: str, enhanced_prompt: str, rag_context: str, tracker: StepTimer) -> tuple[str, list[str]]:
    system_base = "You are a helpful expert assistant."
    if rag_context:
        system_base += f"\n\n{rag_context}"

    if intent == "general":
        tracker.start("llm_general")
        answer = chat(
            "general",
            [
                {"role": "system", "content": system_base},
                {"role": "user", "content": enhanced_prompt},
            ],
        )
        tracker.end("llm_general")
        return answer, ["general (llama3.1:8b)"]

    if intent == "code":
        tracker.start("llm_code")
        answer = chat(
            "coding",
            [
                {
                    "role": "system",
                    "content": system_base + "\nWrite clean, production-ready code with comments.",
                },
                {"role": "user", "content": enhanced_prompt},
            ],
        )
        tracker.end("llm_code")
        return answer, ["coding (qwen2.5-coder:7b)"]

    if intent == "reasoning":
        tracker.start("llm_reasoning")
        answer = chat(
            "reasoning",
            [
                {"role": "system", "content": system_base + "\nThink step by step. Be thorough."},
                {"role": "user", "content": enhanced_prompt},
            ],
        )
        tracker.end("llm_reasoning")
        return answer, ["reasoning (deepseek-r1:8b)"]

    if intent == "code+reasoning":
        tracker.start("llm_reasoning")
        reasoning_answer = chat(
            "reasoning",
            [
                {
                    "role": "system",
                    "content": system_base + "\nThink step by step. Produce a clear solution design.",
                },
                {"role": "user", "content": enhanced_prompt},
            ],
        )
        tracker.end("llm_reasoning")

        tracker.start("llm_code")
        coding_prompt = (
            f"Based on this solution design:\n\n{reasoning_answer}\n\n"
            f"Now implement the code for:\n{enhanced_prompt}"
        )
        code_answer = chat(
            "coding",
            [
                {
                    "role": "system",
                    "content": system_base + "\nWrite clean, production-ready code with comments.",
                },
                {"role": "user", "content": coding_prompt},
            ],
        )
        tracker.end("llm_code")

        combined = f"{reasoning_answer}\n\n---\n\n### Implementation\n\n{code_answer}"
        return combined, ["reasoning (deepseek-r1:8b)", "coding (qwen2.5-coder:7b)"]

    # fallback
    tracker.start("llm_fallback")
    answer = chat(
        "general",
        [
            {"role": "system", "content": system_base},
            {"role": "user", "content": enhanced_prompt},
        ],
    )
    tracker.end("llm_fallback")
    return answer, ["general (llama3.1:8b)"]


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    wants_stream = body.get("stream", False)

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    wants_trace = any(t in user_msg.lower() for t in TRACE_TRIGGERS)
    tracker = StepTimer()

    # Step 1 — parallel enhance+classify AND early RAG keyword check
    msg_lower = user_msg.lower()
    keyword_rag = any(t in msg_lower for t in RAG_TRIGGERS)

    tracker.start("enhance_classify")
    tracker.start("rag")

    rag_future = None
    classify_future = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Always start classify
        classify_future = executor.submit(enhance_and_classify, user_msg)

        # Start RAG early if keyword match detected
        if keyword_rag:
            rag_future = executor.submit(rag_query, user_msg)

        # Get classify result
        try:
            meta = classify_future.result()
        except Exception as e:
            print(f"[Classify] error: {e}")
            meta = {"enhanced_prompt": user_msg, "intent": "general", "needs_rag": False, "rag_reason": ""}

        tracker.end("enhance_classify")

        enhanced_prompt = meta.get("enhanced_prompt", user_msg)
        intent = meta.get("intent", "general")
        needs_rag = meta.get("needs_rag", False)

        # If RAG wasn't started but intent needs it, start now
        rag_context = ""
        if rag_future is None and (needs_rag or intent in ("code", "code+reasoning")):
            rag_future = executor.submit(rag_query, enhanced_prompt)

        # Get RAG result if running
        if rag_future is not None:
            try:
                rag_context = rag_future.result()
            except Exception as e:
                print(f"[RAG] parallel error: {e}")
                rag_context = ""

        tracker.end("rag")

    system_base = "You are a helpful expert assistant."
    if rag_context:
        system_base += f"\n\n{rag_context}"

    # Step 3 — check if task needs decomposition
    if intent == "code+reasoning" and not wants_trace:
        print("[Decomposer] Routing to task decomposer...")
        tracker.start("task_decomposition")
        # Run decomposer in a worker thread so the async event loop isn't blocked.
        # Use asyncio.to_thread (clean, no manual executor management).
        import asyncio

        decomp = await asyncio.to_thread(decompose_and_execute, enhanced_prompt, rag_context)
        tracker.end("task_decomposition")

        answer = decomp["answer"]
        models_used = decomp["models_used"]
        subtask_info = f"\n\n*Decomposed into {len(decomp['subtasks'])} subtasks*"
        answer += subtask_info

        return JSONResponse(
            {
                "id": f"jarvis-{int(tracker.wall_start)}",
                "object": "chat.completion",
                "created": int(tracker.wall_start),
                "model": "jarvis",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    # Step 3 — streaming path (simple intents only, no trace, no chaining)
    if wants_stream and not wants_trace and intent != "code+reasoning":
        model_alias = MODELS.get(intent, "general")
        system_msg = system_base
        if intent == "code":
            system_msg += "\nWrite clean, production-ready code with comments."
        elif intent == "reasoning":
            system_msg += "\nThink step by step. Be thorough."

        msg_id = f"jarvis-{int(time.time())}"

        def stream_generator():
            for token in chat_stream(
                model_alias,
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": enhanced_prompt},
                ],
            ):
                chunk = {
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "jarvis",
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Step 3 — non-streaming execute
    answer, models_used = execute(intent, enhanced_prompt, rag_context, tracker)

    # Step 4 — trace
    if wants_trace:
        trace = (
            f"\n\n---\n**Model trace:**\n"
            f"- Intent: `{intent}`\n"
            f"- RAG used: `{'yes' if rag_context else 'no'}`\n"
            f"- Models: {', '.join(f'`{m}`' for m in models_used)}\n"
            f"- Enhanced prompt: _{enhanced_prompt}_\n\n" + tracker.summary()
        )
        answer += trace

    return JSONResponse(
        {
            "id": f"jarvis-{int(tracker.wall_start)}",
            "object": "chat.completion",
            "created": int(tracker.wall_start),
            "model": "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


# ── Other endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "gateway": "jarvis", "port": 4001}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "coding", "object": "model", "owned_by": "jarvis"},
            {"id": "reasoning", "object": "model", "owned_by": "jarvis"},
            {"id": "general", "object": "model", "owned_by": "jarvis"},
        ],
    }


@app.get("/v1/openapi.json")
def openapi_spec():
    return {"openapi": "3.0.0", "info": {"title": "Jarvis Gateway", "version": "1.0.0"}, "paths": {}}


@app.get("/v1/api/config")
def api_config():
    return {"status": "ok", "name": "Jarvis Gateway", "version": "1.0.0"}


if __name__ == "__main__":
    print("🧠 Jarvis Smart Gateway starting on port 4001...")
    print("   Enhance → Classify → RAG → Route → Stream")
    print("   Point your tools to: http://localhost:4001/v1")
    uvicorn.run(app, host="0.0.0.0", port=4001)
