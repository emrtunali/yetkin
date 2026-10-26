import json
from pathlib import Path

import pytest

from yetkin.baseline import Lexicon
from yetkin.data import Sentence, Span, bio_to_spans, chunks, parse, spans_to_bio
from yetkin.generative import align, messages, parse_answer, target
from yetkin.io import load_predictions, save_predictions
from yetkin.metrics import bootstrap, paired, score

FIX = Path(__file__).parent / "fixtures" / "sample.jsonl"


def sample() -> list[Sentence]:
    return [parse(json.loads(line)) for line in FIX.read_text().splitlines()]


def test_bio_round_trip():
    tags = ["O", "B", "I", "O", "B", "B", "I", "I"]
    spans = bio_to_spans(tags, "skill")
    assert [(s.start, s.end) for s in spans] == [(1, 3), (4, 5), (5, 8)]
    assert spans_to_bio(spans, len(tags), "skill") == tags


def test_stray_inside_tag_starts_a_span():
    assert [(s.start, s.end) for s in bio_to_spans(["O", "I", "I", "O"], "skill")] == [(1, 3)]


def test_layers_are_kept_apart():
    row = {
        "tokens": ["strong", "Python", "skills"],
        "tags_skill": ["B", "I", "I"],
        "tags_knowledge": ["O", "B", "O"],
        "idx": 3,
        "source": "tech",
    }
    s = parse(row)
    assert set(s.spans) == {Span("skill", 0, 3), Span("knowledge", 1, 2)}
    assert s.posting == "tech-3"


def test_scores_strict_and_loose():
    gold = [[Span("skill", 0, 3), Span("knowledge", 4, 5)]]
    pred = [[Span("skill", 1, 3), Span("knowledge", 4, 5), Span("knowledge", 7, 8)]]
    s = score(gold, pred)
    assert s["strict_skill"]["f1"] == 0
    assert s["loose_skill"]["f1"] == 1
    assert s["strict_knowledge"] == {"precision": 0.5, "recall": 1.0, "f1": pytest.approx(2 / 3)}
    assert s["strict_micro"]["precision"] == pytest.approx(1 / 3)


def test_loose_match_counts_each_gold_span_once():
    gold = [[Span("skill", 0, 4)]]
    pred = [[Span("skill", 0, 2), Span("skill", 2, 4)]]
    assert score(gold, pred)["loose_skill"]["precision"] == 0.5


def test_bootstrap_brackets_the_score_and_paired_detects_a_better_system():
    sents = sample()
    gold = [s.spans for s in sents]
    posts = [f"p{i // 4}" for i in range(len(sents))]
    lo, hi = bootstrap(gold, gold, posts, n=200)
    assert lo == hi == 1.0
    empty = [[] for _ in gold]
    r = paired(gold, gold, empty, posts, n=200)
    assert r["diff"] == 1.0 and r["low"] > 0 and r["p"] == 0


def test_generative_target_parse_align_round_trip():
    for s in sample():
        assert set(align(s.tokens, parse_answer(target(s)))) == set(s.spans) or _has_repeats(s)


def _has_repeats(s: Sentence) -> bool:
    texts = [(sp.type, sp.text(s.tokens).lower()) for sp in s.spans]
    return len(texts) != len(set(texts)) or any(s.tokens.count(t) > 1 for t in s.tokens)


def test_parse_answer_is_forgiving():
    assert parse_answer('```json\n{"skills": ["code review"], "knowledge": []}\n```') == {
        "skills": ["code review"],
        "knowledge": [],
    }
    assert parse_answer('Sure! Here you go: {"skills": [], "knowledge": ["SQL"]} hope it helps') == {
        "skills": [],
        "knowledge": ["SQL"],
    }
    assert parse_answer("no json here") == {"skills": [], "knowledge": []}
    assert parse_answer('{"skills": "teamwork"}') == {"skills": ["teamwork"], "knowledge": []}
    assert parse_answer('{"skills": 3, "knowledge": null}') == {"skills": [], "knowledge": []}


def test_align_drops_hallucinations_and_handles_repeats():
    toks = ["Python", "and", "SQL", ",", "more", "Python"]
    spans = align(toks, {"skills": [], "knowledge": ["python", "Python", "Rust"]})
    assert spans == [Span("knowledge", 0, 1), Span("knowledge", 5, 6)]


def test_messages_include_shots_in_order():
    s = sample()
    m = messages(s[0], shots=s[1:3])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_chunks_never_cut_a_span():
    s = Sentence([f"w{i}" for i in range(300)], [Span("skill", 125, 135), Span("knowledge", 250, 252)])
    parts = chunks(s, 128)
    assert sum(len(p.tokens) for p in parts) == 300
    assert sum(len(p.spans) for p in parts) == 2
    assert all(0 <= sp.start < sp.end <= len(p.tokens) for p in parts for sp in p.spans)


def test_lexicon_baseline_finds_seen_spans():
    sents = sample()
    lex = Lexicon(sents)
    pred = [lex.predict(s.tokens) for s in sents]
    assert score([s.spans for s in sents], pred)["loose_micro"]["recall"] > 0.8


def test_predictions_file_round_trip(tmp_path):
    sents = sample()[:5]
    pred = [s.spans for s in sents]
    save_predictions(tmp_path / "p.jsonl", sents, pred, meta={"name": "gold"})
    back, meta = load_predictions(tmp_path / "p.jsonl")
    assert back == pred and meta == {"name": "gold"}
