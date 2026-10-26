"""Span extraction as text generation, for instruction-tuned LLMs.

The model reads one sentence and answers with JSON: {"skills": [...], "knowledge": [...]}, each
item copied verbatim from the sentence. ``align`` maps the strings back onto tokens so a generated
answer is scored exactly like a tagger's output. Strings that do not occur in the sentence are
dropped (a hallucinated skill counts as a miss, never as a hit).
"""

from __future__ import annotations

import json
import re

from .data import Sentence, Span

SYSTEM = (
    "You extract skills and knowledge from one sentence of a job posting.\n"
    'Skills are things a person can do (e.g. "manage stakeholders", "write clean code").\n'
    "Knowledge is what a person knows: tools, languages, fields, certificates\n"
    '(e.g. "Python", "AWS", "accounting").\n'
    "Copy each span exactly as it appears in the sentence. If there are none, return empty lists.\n"
    'Answer only with JSON: {"skills": [...], "knowledge": [...]}'
)

KEYS = {"skills": "skill", "knowledge": "knowledge"}


def target(s: Sentence) -> str:
    """The answer an extractor should give for a sentence."""
    return json.dumps(
        {
            "skills": [sp.text(s.tokens) for sp in s.of("skill")],
            "knowledge": [sp.text(s.tokens) for sp in s.of("knowledge")],
        },
        ensure_ascii=False,
    )


def messages(s: Sentence, shots: list[Sentence] = ()) -> list[dict]:
    """Chat messages: instructions, optional worked examples, then the sentence."""
    out = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        out += [
            {"role": "user", "content": " ".join(ex.tokens)},
            {"role": "assistant", "content": target(ex)},
        ]
    out.append({"role": "user", "content": " ".join(s.tokens)})
    return out


def parse_answer(text: str) -> dict[str, list[str]]:
    """Read the JSON object out of a model's answer, tolerating code fences and chatter."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"skills": [], "knowledge": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # last resort: cut at the first closing brace
        try:
            obj = json.loads(m.group(0)[: m.group(0).index("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return {"skills": [], "knowledge": []}
    out = {}
    for k in KEYS:
        v = obj.get(k, []) if isinstance(obj, dict) else []
        if isinstance(v, str):  # a single answer written without a list
            v = [v]
        elif not isinstance(v, list):
            v = []
        out[k] = [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return out


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def align(tokens: list[str], answer: dict[str, list[str]]) -> list[Span]:
    """Turn extracted strings into token spans: each string takes its first unused exact match
    (case-insensitive, whitespace-normalised). Unfound strings are dropped."""
    low = [_norm(t) for t in tokens]
    used: set[tuple[str, int, int]] = set()
    spans = []
    for key, type_ in KEYS.items():
        for item in answer.get(key, []):
            words = _norm(item).split(" ")
            n = len(words)
            for i in range(len(low) - n + 1):
                if low[i : i + n] == words and (type_, i, i + n) not in used:
                    used.add((type_, i, i + n))
                    spans.append(Span(type_, i, i + n))
                    break
    return spans
