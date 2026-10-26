"""Span-level evaluation with uncertainty.

Strict match (same type, same first and last token) is the SkillSpan paper's metric. Loose match
(same type, any token overlap) shows how much of the gap is boundary disagreement rather than a
missed skill. Confidence intervals resample whole postings, because sentences of one posting are
not independent; the paired version tells whether one system beats another beyond noise.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .data import TYPES, Span


def _overlap(a: Span, b: Span) -> bool:
    return a.type == b.type and a.start < b.end and b.start < a.end


def counts(
    gold: Sequence[Sequence[Span]], pred: Sequence[Sequence[Span]], type_: str, loose: bool = False
) -> tuple[int, int, int]:
    """(true positives, predicted, gold) over sentences, for one span type."""
    tp = n_pred = n_gold = 0
    for g, p in zip(gold, pred, strict=True):
        g = [s for s in g if s.type == type_]
        p = [s for s in p if s.type == type_]
        n_gold += len(g)
        n_pred += len(p)
        if loose:
            used: set[int] = set()
            for s in p:
                hit = next((i for i, t in enumerate(g) if i not in used and _overlap(s, t)), None)
                if hit is not None:
                    used.add(hit)
                    tp += 1
        else:
            tp += len(set(p) & set(g))
    return tp, n_pred, n_gold


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def score(gold: Sequence[Sequence[Span]], pred: Sequence[Sequence[Span]]) -> dict[str, dict[str, float]]:
    """Strict and loose precision/recall/F1 per type, and micro-averaged over both types."""
    out = {}
    for mode in ("strict", "loose"):
        tot = [0, 0, 0]
        for t in TYPES:
            c = counts(gold, pred, t, loose=mode == "loose")
            out[f"{mode}_{t}"] = prf(*c)
            tot = [a + b for a, b in zip(tot, c, strict=True)]
        out[f"{mode}_micro"] = prf(*tot)
    return out


def _by_posting(postings: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(postings):
        groups.setdefault(p, []).append(i)
    return groups


def bootstrap(
    gold, pred, postings: Sequence[str], key: str = "strict_micro", n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % interval of an F1 by resampling postings with replacement."""
    groups = list(_by_posting(postings).values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [i for g in rng.choices(groups, k=len(groups)) for i in g]
        vals.append(score([gold[i] for i in idx], [pred[i] for i in idx])[key]["f1"])
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def paired(
    gold, pred_a, pred_b, postings: Sequence[str], key: str = "strict_micro", n: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap: the F1 difference A − B, its 95 % interval and how often B is not worse."""
    groups = list(_by_posting(postings).values())
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = [i for g in rng.choices(groups, k=len(groups)) for i in g]
        g = [gold[i] for i in idx]
        diffs.append(
            score(g, [pred_a[i] for i in idx])[key]["f1"] - score(g, [pred_b[i] for i in idx])[key]["f1"]
        )
    diffs.sort()
    full = score(gold, pred_a)[key]["f1"] - score(gold, pred_b)[key]["f1"]
    return {
        "diff": full,
        "low": diffs[int(0.025 * n)],
        "high": diffs[int(0.975 * n) - 1],
        "p": sum(d <= 0 for d in diffs) / n,
    }
