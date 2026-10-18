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


# Thinking the gateway streams as <think>, or that Open WebUI stores as a reasoning <details> block.
_THINK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL)
_REASONING_DETAILS = re.compile(r'<details\s+type="reasoning"[^>]*>.*?</details>', re.DOTALL)


def clean_assistant(text: str) -> str:
    """An assistant turn as the model should see it next time: without its thinking."""
    text = _REASONING_DETAILS.sub("", _THINK.sub("", text))
    return text.strip()
