"""A dictionary baseline with no learning: every span seen in training, matched longest first.

It sets the floor a model has to beat, and shows how much of the task is memorising a list of
skill names versus understanding context ("design" as a skill vs as a noun).
"""

from __future__ import annotations

from collections import Counter

from .data import TYPES, Sentence, Span


class Lexicon:
    def __init__(self, train: list[Sentence], min_count: int = 1):
        seen: dict[str, Counter] = {}
        for s in train:
            for sp in s.spans:
                key = sp.text(s.tokens).lower()
                seen.setdefault(key, Counter())[sp.type] += 1
        # each phrase keeps its most frequent type
        self.entries = {k: c.most_common(1)[0][0] for k, c in seen.items() if sum(c.values()) >= min_count}
        self.max_len = max((len(k.split()) for k in self.entries), default=1)

    def predict(self, tokens: list[str]) -> list[Span]:
        low = [t.lower() for t in tokens]
        out, i = [], 0
        while i < len(low):
            for n in range(min(self.max_len, len(low) - i), 0, -1):
                key = " ".join(low[i : i + n])
                if key in self.entries:
                    out.append(Span(self.entries[key], i, i + n))
                    i += n
                    break
            else:
                i += 1
        return [s for s in out if s.type in TYPES]


if __name__ == "__main__":
    from pathlib import Path

    from .data import load
    from .io import save_predictions

    train, test = load("train"), load("test")
    lex = Lexicon(train)
    out = Path(__file__).resolve().parents[2] / "results" / "predictions" / "lexicon.jsonl"
    save_predictions(
        out,
        test,
        [lex.predict(s.tokens) for s in test],
        meta={"name": "Lexicon (no learning)", "entries": len(lex.entries)},
    )
    print(f"{len(lex.entries)} phrases -> {out}")
