"""Model traces: remember the last few, and spot a follow-up that asks for the previous one."""

import hashlib
import re
import threading
from collections import OrderedDict

from jarvis.messages import clean_assistant, text_of

# Phrases that ask how the PREVIOUS answer was produced. Kept narrow on purpose:
# "which model should I use for X" and "why did you pick Postgres?" are real questions.
TRACE_FOLLOWUPS = (
    "show trace",
    "show the trace",
    "model trace",
    "which model answered",
    "what model answered",
    "which model did you use",
    "what model did you use",
)


class TraceStore:
    """The last `size` traces, keyed by a hash of the answer text the client will send back."""

    def __init__(self, size: int = 256):
        self.size = size
        self._data: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()  # answers are stored from worker threads while requests read

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def key(answer: str) -> str | None:
        """Hash of the cleaned, whitespace-normalized answer. None if nothing is left to hash."""
        text = " ".join(clean_assistant(answer).split())
        return hashlib.sha256(text.encode()).hexdigest() if text else None

    def put(self, answer: str, trace: str) -> None:
        key = self.key(answer)
        if key is None:
            return
        with self._lock:
            self._data[key] = trace
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def get(self, answer: str) -> str | None:
        key = self.key(answer)
        if not key:
            return None
        with self._lock:
            return self._data.get(key)


MAX_FOLLOWUP_WORDS = 8
_NOT_WORD = re.compile(r"[^\w']+")


def asks_for_trace(text: str) -> bool:
    """A short message that starts or ends with a trace phrase. Longer ones are real questions."""
    ask = " ".join(_NOT_WORD.sub(" ", text.lower()).split())
    if not ask or len(ask.split()) > MAX_FOLLOWUP_WORDS:
        return False
    return any(ask.startswith(phrase) or ask.endswith(phrase) for phrase in TRACE_FOLLOWUPS)


def trace_followup(messages: list[dict], store: TraceStore) -> str | None:
    """If the latest user message asks for a trace and one is stored for the answer before it, return it."""
    end = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), -1)
    if end < 0 or not asks_for_trace(text_of(messages[end].get("content"))):
        return None
    for message in reversed(messages[:end]):
        if message.get("role") == "assistant":
            return store.get(text_of(message.get("content")))
    return None
