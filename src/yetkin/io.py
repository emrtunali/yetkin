"""Predictions on disk: one JSON line per test sentence, so every system is scored the same way."""

from __future__ import annotations

import json
from pathlib import Path

from .data import Sentence, Span


def save_predictions(
    path: str | Path,
    sentences: list[Sentence],
    pred: list[list[Span]],
    raw: list[str] | None = None,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if meta:
            f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for i, (s, p) in enumerate(zip(sentences, pred, strict=True)):
            row = {"posting": s.posting, "tokens": s.tokens, "pred": [[x.type, x.start, x.end] for x in p]}
            if raw is not None:
                row["raw"] = raw[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions(path: str | Path) -> tuple[list[list[Span]], dict]:
    pred, meta = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            pred.append([Span(t, a, b) for t, a, b in row["pred"]])
    return pred, meta
