"""From a raw job posting to model input, and from model output to a skills report.

SkillSpan is tokenised on whitespace with punctuation split off, one sentence per line, so a pasted
posting is cut into sentences and tokens the same way before tagging.
"""

from __future__ import annotations

import re
from collections import Counter

from .data import Sentence, Span

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./-]*[A-Za-z0-9+#]|[A-Za-z0-9]|[^\sA-Za-z0-9]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•\-*])|\n+")


def sentences(text: str) -> list[Sentence]:
    """Split lines and sentences, strip bullets and numbering, keep tokens like "C++" and "Node.js" whole."""
    out = []
    for part in SENTENCE_END.split(text):
        part = re.sub(r"^\s*([•\-*·]|\d+[.)])\s*", "", part).strip()
        if part:
            out.append(Sentence(TOKEN.findall(part)))
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .,;:")


def report(sents: list[Sentence], pred: list[list[Span]]) -> dict[str, list[tuple[str, int]]]:
    """Distinct skills and knowledge with how often each was mentioned, most frequent first."""
    by = {"skill": Counter(), "knowledge": Counter()}
    shown: dict[str, str] = {}
    for s, spans in zip(sents, pred, strict=True):
        for sp in spans:
            text = sp.text(s.tokens)
            key = norm(text)
            shown.setdefault(key, text)
            by[sp.type][key] += 1
    return {t: [(shown[k], n) for k, n in c.most_common()] for t, c in by.items()}


def gap(found: dict[str, list[tuple[str, int]]], mine: str) -> dict[str, list[str]]:
    """Which of the posting's skills you listed, and which you did not. Matching is lenient:
    "python" matches "Python 3", "aws" matches "AWS cloud"."""
    have = [norm(x) for x in re.split(r"[,\n;]+", mine) if x.strip()]
    asked = [(text, norm(text)) for t in ("knowledge", "skill") for text, _ in found[t]]
    match, missing = [], []
    for text, key in asked:
        hit = any(
            h == key or re.search(rf"\b{re.escape(h)}\b", key) or re.search(rf"\b{re.escape(key)}\b", h)
            for h in have
        )
        (match if hit else missing).append(text)
    return {"match": match, "missing": missing}
