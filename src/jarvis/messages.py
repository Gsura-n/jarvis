"""Helpers for OpenAI-style chat messages."""


def text_of(content) -> str:
    """OpenAI content can be a string or a list of parts (Continue sends parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""
