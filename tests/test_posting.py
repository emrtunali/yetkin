from yetkin.data import Span
from yetkin.posting import gap, report, sentences


def test_sentences_split_lines_bullets_and_keep_tech_tokens():
    s = sentences("Requirements:\n- 3+ years with C++ and Node.js.\n• Strong SQL skills. Team player")
    assert [x.tokens for x in s] == [
        ["Requirements", ":"],
        ["3+", "years", "with", "C++", "and", "Node.js", "."],
        ["Strong", "SQL", "skills", "."],
        ["Team", "player"],
    ]


def test_report_counts_distinct_mentions():
    s = sentences("Python and SQL.\nMore Python")
    pred = [[Span("knowledge", 0, 1), Span("knowledge", 2, 3)], [Span("knowledge", 1, 2)]]
    assert report(s, pred) == {"skill": [], "knowledge": [("Python", 2), ("SQL", 1)]}


def test_gap_is_lenient_but_not_sloppy():
    found = {
        "knowledge": [("Python 3", 1), ("AWS", 1), ("Kubernetes", 1)],
        "skill": [("mentoring junior engineers", 1)],
    }
    g = gap(found, "python, aws cloud, mentoring")
    assert g["match"] == ["Python 3", "AWS", "mentoring junior engineers"]
    assert g["missing"] == ["Kubernetes"]
    assert gap(found, "go")["match"] == []  # "go" must not match inside other words
