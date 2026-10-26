"""Score every system's predictions on the test set and write results/RESULTS.md.

    python -m yetkin.compare results/predictions/*.jsonl

For each system: strict and loose F1 per type with 95 % intervals over postings, and a paired
bootstrap against the best system. Also a chart, results/f1.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load
from .io import load_predictions
from .metrics import bootstrap, paired, score

ROOT = Path(__file__).resolve().parents[2]


def compare(files: list[Path], out: Path = ROOT / "results", n_boot: int = 1000) -> list[dict]:
    test = load("test")
    gold = [s.spans for s in test]
    posts = [s.posting for s in test]
    systems = []
    for f in files:
        pred, meta = load_predictions(f)
        if len(pred) != len(gold):
            raise ValueError(f"{f.name}: {len(pred)} predictions for {len(gold)} test sentences")
        sc = score(gold, pred)
        lo, hi = bootstrap(gold, pred, posts, n=n_boot)
        systems.append(
            {
                "name": meta.get("name", f.stem),
                "file": f.name,
                "meta": meta,
                "score": sc,
                "ci": [lo, hi],
                "pred": pred,
            }
        )
    systems.sort(key=lambda x: -x["score"]["strict_micro"]["f1"])
    best = systems[0]
    for s in systems[1:]:
        s["vs_best"] = paired(gold, best["pred"], s["pred"], posts, n=n_boot)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "pred"} for s in systems], indent=1)
    )
    write_markdown(systems, out / "RESULTS.md")
    chart(systems, out / "f1.png")
    return systems


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def write_markdown(systems: list[dict], path: Path) -> None:
    lines = [
        "# Results",
        "",
        "SkillSpan test set: 3,569 sentences from 65 job postings. Span F1 in %; strict = exact boundaries,",
        "loose = any overlap. Intervals: 95 % bootstrap over postings. The last column is a paired bootstrap",
        "against the best system (difference in strict micro F1, and its interval).",
        "",
        "| System | Strict F1 (micro) | 95 % interval | Skill | Knowledge | Loose F1 | vs best |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for s in systems:
        sc = s["score"]
        vs = (
            "—"
            if "vs_best" not in s
            else f"{pct(-s['vs_best']['diff'])} [{pct(-s['vs_best']['high'])}, {pct(-s['vs_best']['low'])}]"
        )
        lines.append(
            f"| {s['name']} | **{pct(sc['strict_micro']['f1'])}** | {pct(s['ci'][0])}–{pct(s['ci'][1])} | "
            f"{pct(sc['strict_skill']['f1'])} | "
            f"{pct(sc['strict_knowledge']['f1'])} | {pct(sc['loose_micro']['f1'])} | {vs} |"
        )
    # runs of one system with different seeds: mean and spread
    groups: dict[str, list[dict]] = {}
    for s in systems:
        groups.setdefault(s["meta"].get("system", s["name"]), []).append(s)
    if any(len(g) > 1 for g in groups.values()):
        import statistics as st

        lines += [
            "",
            "## Across seeds",
            "",
            "| System | Runs | Strict F1 mean | SD | Skill | Knowledge |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, g in sorted(
            groups.items(), key=lambda kv: -st.mean(x["score"]["strict_micro"]["f1"] for x in kv[1])
        ):
            f = [x["score"]["strict_micro"]["f1"] for x in g]
            sk = st.mean(x["score"]["strict_skill"]["f1"] for x in g)
            kn = st.mean(x["score"]["strict_knowledge"]["f1"] for x in g)
            sd = st.stdev(f) if len(f) > 1 else 0.0
            lines.append(f"| {name} | {len(g)} | **{pct(st.mean(f))}** | {pct(sd)} | {pct(sk)} | {pct(kn)} |")
    lines += ["", "## Settings", ""]
    for s in systems:
        meta = {k: v for k, v in s["meta"].items() if k not in ("name",)}
        lines.append(f"- **{s['name']}**: " + ", ".join(f"{k} = {v}" for k, v in meta.items()))
    path.write_text("\n".join(lines) + "\n")


def chart(systems: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [s["name"] for s in systems][::-1]
    f1 = [100 * s["score"]["strict_micro"]["f1"] for s in systems][::-1]
    loose = [100 * s["score"]["loose_micro"]["f1"] for s in systems][::-1]
    lo = [100 * s["ci"][0] for s in systems][::-1]
    hi = [100 * s["ci"][1] for s in systems][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.4), dpi=150)
    y = range(len(names))
    ax.barh(y, loose, color="#c9d6ea", height=0.55, label="loose (any overlap)")
    ax.barh(y, f1, color="#2a78d6", height=0.55, label="strict (exact span)")
    ax.errorbar(
        f1,
        y,
        xerr=[[a - b for a, b in zip(f1, lo, strict=True)], [b - a for a, b in zip(f1, hi, strict=True)]],
        fmt="none",
        ecolor="#0b0d14",
        capsize=3,
        lw=1,
    )
    for i, v in enumerate(f1):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(list(y), names, fontsize=9)
    ax.set_xlabel("span F1 on the SkillSpan test set (%)")
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    for s in compare(a.files):
        print(
            f"{s['name']:40s} strict F1 {pct(s['score']['strict_micro']['f1'])}  "
            f"[{pct(s['ci'][0])}, {pct(s['ci'][1])}]"
        )
