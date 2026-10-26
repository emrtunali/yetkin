"""SkillSpan: job postings with skill and knowledge spans annotated token by token.

Zhang, Jensen, Sonniks and Plank (2022), "SkillSpan: Hard and Soft Skill Extraction from English
Job Postings", NAACL. CC BY 4.0. Each line is one sentence of a posting, with two independent BIO
layers: ``tags_skill`` and ``tags_knowledge``. The two layers may overlap, which is why they are
kept apart here instead of being merged into one tag set.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

URL = "https://huggingface.co/datasets/jjzha/skillspan/resolve/main/{split}.json"
TYPES = ("skill", "knowledge")
CACHE = Path(os.environ.get("YETKIN_CACHE", Path.home() / ".cache" / "yetkin"))


@dataclass(frozen=True)
class Span:
    type: str  # "skill" | "knowledge"
    start: int  # first token
    end: int  # one past the last token

    def text(self, tokens: list[str]) -> str:
        return " ".join(tokens[self.start : self.end])


@dataclass
class Sentence:
    tokens: list[str]
    spans: list[Span] = field(default_factory=list)
    posting: str = ""  # "<source>-<idx>": sentences of one posting share it
    source: str = ""

    def of(self, type_: str) -> list[Span]:
        return [s for s in self.spans if s.type == type_]


def bio_to_spans(tags: list[str], type_: str) -> list[Span]:
    """BIO tags -> spans. A stray "I" without a "B" starts a span (as conlleval does)."""
    spans, start = [], None
    for i, t in enumerate([*tags, "O"]):
        t = t.split("-")[0]
        if t == "I" and start is not None:
            continue  # the span goes on
        if start is not None:
            spans.append(Span(type_, start, i))
            start = None
        if t in ("B", "I"):
            start = i
    return spans


def spans_to_bio(spans: list[Span], n: int, type_: str) -> list[str]:
    tags = ["O"] * n
    for s in spans:
        if s.type != type_:
            continue
        tags[s.start] = "B"
        for i in range(s.start + 1, s.end):
            tags[i] = "I"
    return tags


def parse(row: dict) -> Sentence:
    spans = [sp for t in TYPES for sp in bio_to_spans(row[f"tags_{t}"], t)]
    return Sentence(
        row["tokens"], spans, f"{row.get('source', '')}-{row.get('idx', '')}", row.get("source", "")
    )


def load(split: str, path: str | Path | None = None) -> list[Sentence]:
    """Sentences of ``split`` ("train" | "dev" | "test"), downloaded once and cached."""
    if path is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"skillspan-{split}.json"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(split=split), path)
    with open(path, encoding="utf-8") as f:
        return [parse(json.loads(line)) for line in f if line.strip()]


def chunks(sentence: Sentence, max_tokens: int = 128) -> list[Sentence]:
    """Split very long sentences (a few in train run to hundreds of tokens) without cutting spans."""
    if len(sentence.tokens) <= max_tokens:
        return [sentence]
    out, start = [], 0
    while start < len(sentence.tokens):
        end = min(start + max_tokens, len(sentence.tokens))
        # move the cut left until it falls between spans
        while end < len(sentence.tokens) and any(s.start < end < s.end for s in sentence.spans):
            end -= 1
        if end <= start:  # a single span longer than the window: keep it whole
            end = max(s.end for s in sentence.spans if s.start <= start < s.end)
        spans = [
            Span(s.type, s.start - start, s.end - start)
            for s in sentence.spans
            if start <= s.start and s.end <= end
        ]
        out.append(Sentence(sentence.tokens[start:end], spans, sentence.posting, sentence.source))
        start = end
    return out
