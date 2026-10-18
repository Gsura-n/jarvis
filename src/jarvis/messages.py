"""Helpers for OpenAI-style chat messages."""

import re


def text_of(content) -> str:
    """OpenAI content can be a string or a list of parts (Continue sends parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p["text"]
            for p in content
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
    return ""


# Starts the "Model trace" appendix. The code that writes a trace and the code that strips it share this.
TRACE_MARK = "\n\n---\n**Model trace:**"

# Thinking the gateway streams as <think>, or that Open WebUI stores as a reasoning <details> block.
_THINK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL)
_UNCLOSED_THINK = re.compile(r"\s*<(think|thinking)>")
_REASONING_DETAILS = re.compile(r'<details\s+type="reasoning"[^>]*>.*?</details>', re.DOTALL)


def clean_assistant(text: str) -> str:
    """An assistant turn as the model should see it next time: without its thinking or trace.

    A turn that still starts with an unclosed <think> was stopped mid-thinking, so none of
    it is an answer. A <think> later in normal text is left alone."""
    text = _REASONING_DETAILS.sub("", _THINK.sub("", text))
    cut = text.find(TRACE_MARK.strip())
    text = (text if cut < 0 else text[:cut]).strip()
    return "" if _UNCLOSED_THINK.match(text) else text


def system_text(messages: list) -> str:
    """The client's own system messages, joined. Empty if there are none."""
    texts = (text_of(m.get("content")) for m in messages if m.get("role") == "system")
    return "\n\n".join(t for t in texts if t)
