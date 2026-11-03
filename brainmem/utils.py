"""Small helpers shared by the memory stores."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Any, Iterable, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "it", "to", "of", "on", "in", "and", "then", "with", "into",
    "onto", "at", "is", "be", "as", "for", "from", "up", "down", "that", "this",
    "its", "so", "by", "or", "you", "your", "please",
}


def tokens(text: str) -> Set[str]:
    """Lower-cased content tokens of a string."""
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def short_hash(payload: Any, n: int = 12) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:n]


def canonical_object(name: str) -> str:
    """'Fork|+00.1|...' -> 'fork', 'Cabinet_2' -> 'cabinet', 'CounterTop' -> 'countertop'."""
    base = (name or "").split("|")[0]
    base = re.sub(r"_\d+$", "", base)
    return base.strip().lower()


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON so that an interrupted run never leaves a truncated store."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_object(text: str) -> Any:
    """Parse the first JSON object in an LLM response (plain or fenced)."""
    if text is None:
        raise ValueError("empty response")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(s[start:end + 1])
