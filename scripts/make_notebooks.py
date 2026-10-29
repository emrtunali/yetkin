"""Write the Colab notebooks in notebooks/ from the cell lists below.

    python scripts/make_notebooks.py

Keeping the notebooks as code here makes them reviewable in diffs and lets the tests check that
every code cell at least compiles.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "emrtunali/yetkin"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


SETUP = code(f'''
# Runtime → Change runtime type → T4 GPU, then Runtime → Run all.
!pip -q install "yetkin[train,qlora,track] @ git+https://github.com/{REPO}.git"

import json, os, time
import torch
from google.colab import drive, userdata

assert torch.cuda.is_available(), "No GPU: Runtime → Change runtime type → T4 GPU"
print(torch.cuda.get_device_name(0))

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/yetkin"          # results survive a disconnect here
os.makedirs(f"{{OUT}}/predictions", exist_ok=True)

def secret(name):
    """Optional keys from the 🔑 Secrets panel: HF_TOKEN (to publish models), WANDB_API_KEY (to log runs)."""
    try:
        return userdata.get(name)
    except Exception:
        return None
''')

WANDB = code("""
# Experiment tracking: Weights & Biases when a WANDB_API_KEY secret exists, otherwise just printed.
wandb = None
if secret("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = secret("WANDB_API_KEY")
    import wandb
print("W&B logging:", "on" if wandb else "off")
""")

DATA = code("""
from yetkin.data import load, chunks
train = [c for s in load("train") for c in chunks(s, 128)]
dev, test = load("dev"), load("test")
print(f"train {len(train):,} sentences · dev {len(dev):,} · test {len(test):,}")
""")

ENCODER = notebook(
    [
        md(f"""
# 1 · Fine-tuned encoders with two tagging heads

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/01_encoder.ipynb)

Two pre-trained encoders, each fine-tuned with **one BIO head per span type** (skill, knowledge) on a shared
body, three seeds each. JobBERT was further pre-trained on job postings (Zhang et al., 2022); DeBERTa-v3 is a
strong general encoder. The epoch with the best dev F1 is kept; test predictions go to Google Drive.

About 25 minutes on a T4.
"""),
        SETUP,
        WANDB,
        DATA,
        code("""
MODELS = {"JobBERT": ("jjzha/jobbert-base-cased", 3e-5), "DeBERTa-v3-base": ("microsoft/deberta-v3-base", 2e-5)}
SEEDS = [1, 2, 3]
EPOCHS, BATCH = 5, 16
"""),
        code("""
from yetkin import encoder
from yetkin.io import save_predictions
from yetkin.metrics import score

runs = []
for label, (base, lr) in MODELS.items():
    for seed in SEEDS:
        slug = f"{label.lower().replace(' ', '-')}-seed{seed}"
        path = f"{OUT}/predictions/{slug}.jsonl"
        if os.path.exists(path):
            print("done already:", slug); continue
        run = wandb.init(project="yetkin", name=slug, config={"base": base, "lr": lr, "seed": seed, "epochs": EPOCHS}) if wandb else None
        t0 = time.time()
        model, tok, info = encoder.train(base, train, dev, epochs=EPOCHS, batch_size=BATCH, lr=lr, seed=seed,
                                         on_epoch=(lambda h: run.log({"loss": h["loss"], "dev_f1": h["dev_f1"]})) if run else None)
        minutes = (time.time() - t0) / 60
        t1 = time.time()
        pred = encoder.predict(model, tok, test)
        per_1k = (time.time() - t1) / len(test) * 1000
        s = score([x.spans for x in test], pred)
        meta = {"name": f"{label} · seed {seed}", "system": label, "seed": seed, "base": base, "lr": lr, "epochs": EPOCHS,
                "best_dev_f1": round(info["best_dev_f1"], 4), "train_minutes": round(minutes, 1), "seconds_per_1k_sentences": round(per_1k, 2)}
        save_predictions(path, test, pred, meta=meta)
        model.save(f"/content/runs/{slug}", tok)            # local disk; the best one is copied to Drive below
        runs.append({**meta, "test_f1": s["strict_micro"]["f1"]})
        print(f"{slug}: dev {info['best_dev_f1']:.3f}  test {s['strict_micro']['f1']:.3f}  ({minutes:.1f} min)")
        if run: run.log({"test_f1": s["strict_micro"]["f1"]}); run.finish()
        del model; torch.cuda.empty_cache()
"""),
        code("""
# Keep the run with the best *dev* score (choosing on test would be cheating) and publish it if you like.
# Every finished run is read back from Drive, so runs from an earlier session take part too.
import glob, shutil
from yetkin.io import load_predictions
runs = []
for path in sorted(glob.glob(f"{OUT}/predictions/*-seed*.jsonl")):
    pred, meta = load_predictions(path)
    runs.append({**meta, "slug": os.path.basename(path)[:-6], "test_f1": score([x.spans for x in test], pred)["strict_micro"]["f1"]})
    print(f"{meta['name']:<26} dev {meta['best_dev_f1']:.3f}  test {runs[-1]['test_f1']:.3f}")
best = max(runs, key=lambda r: r["best_dev_f1"])
if not os.path.exists(f"/content/runs/{best['slug']}"):
    print(f"The weights of {best['slug']} were on a session that ended: delete its predictions file and rerun.")
else:
    shutil.copytree(f"/content/runs/{best['slug']}", f"{OUT}/models/encoder", dirs_exist_ok=True)
    json.dump(best, open(f"{OUT}/models/encoder/run.json", "w"), indent=1)
    print("best on dev:", best["name"], f"(test {best['test_f1']:.3f})")
    if secret("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi(token=secret("HF_TOKEN"))
        repo = api.create_repo("yetkin-skill-tagger", exist_ok=True).repo_id
        api.upload_folder(folder_path=f"{OUT}/models/encoder", repo_id=repo)
        print("published:", f"https://huggingface.co/{repo}")
"""),
    ]
)

QLORA = notebook(
    [
        md(f"""
# 2 · A small LLM fine-tuned with QLoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/02_qlora.ipynb)

The base model is loaded in **4-bit NF4** and frozen; **LoRA adapters** (rank 16) on every linear layer are trained
to answer with JSON (`{{"skills": [...], "knowledge": [...]}}`). The loss covers only the answer tokens. Answers are
mapped back onto the sentence's tokens, so the score is computed exactly like the taggers'.

About 3 hours on a T4: 2.5 hours of training (610 steps), then generating answers for the 3,569 test sentences. Training is checkpointed to Drive, so after a disconnect *Run all* picks up where it stopped.
"""),
        SETUP,
        WANDB,
        DATA,
        code("""
BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # also worth a run: "Qwen/Qwen2.5-0.5B-Instruct" (a size ablation)
R, ALPHA, EPOCHS, LR = 16, 32, 2, 2e-4
slug = "qlora-" + BASE.split("/")[-1].lower()
"""),
        code("""
from yetkin import llm
adapter = f"{OUT}/models/{slug}"
if not os.path.exists(f"{adapter}/adapter_config.json"):
    run = wandb.init(project="yetkin", name=slug, config={"base": BASE, "r": R, "alpha": ALPHA, "epochs": EPOCHS, "lr": LR}) if wandb else None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    # Checkpoints go to Drive every 50 steps; after a disconnect, Run all continues from the last one.
    info = llm.qlora(BASE, train, adapter, epochs=EPOCHS, r=R, alpha=ALPHA, lr=LR, checkpoint_dir=f"{adapter}.checkpoint",
                     on_step=(lambda h: run.log(h)) if run else None)
    info["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    json.dump({k: v for k, v in info.items() if k != "history"}, open(f"{adapter}/train_info.json", "w"), indent=1)
    if run: run.finish()
    print(f"trained in {info['train_minutes']} min, peak GPU memory {info['peak_gb']} GB")
else:
    print("adapter already trained:", adapter)
"""),
        code("""
from yetkin.io import save_predictions
from yetkin.metrics import score
path = f"{OUT}/predictions/{slug}.jsonl"
if not os.path.exists(path):
    model, tok = llm.load(BASE, four_bit=True, adapter=adapter)
    t0 = time.time()
    pred, raw = llm.extract(model, tok, test, batch_size=32, log=print)
    per_1k = (time.time() - t0) / len(test) * 1000
    info = json.load(open(f"{adapter}/train_info.json"))
    meta = {"name": f"QLoRA · {BASE.split('/')[-1]}", "system": f"QLoRA · {BASE.split('/')[-1]}", "base": BASE, "r": R, "alpha": ALPHA,
            "epochs": EPOCHS, "lr": LR, "trainable_params": info["trainable"], "train_minutes": info["train_minutes"],
            "peak_gpu_gb": info["peak_gb"], "seconds_per_1k_sentences": round(per_1k, 1)}
    save_predictions(path, test, pred, raw=raw, meta=meta)
    print("test strict F1:", round(score([x.spans for x in test], pred)["strict_micro"]["f1"], 4))
"""),
        code("""
# Optional: publish the adapter (a few MB) to the Hugging Face Hub.
if secret("HF_TOKEN"):
    from huggingface_hub import HfApi
    api = HfApi(token=secret("HF_TOKEN"))
    repo = api.create_repo(f"yetkin-{slug}", exist_ok=True).repo_id
    api.upload_folder(folder_path=adapter, repo_id=repo)
    print("published:", f"https://huggingface.co/{repo}")
"""),
    ]
)

COMPARE = notebook(
    [
        md(f"""
# 3 · Compare everything

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/03_compare.ipynb)

Scores every prediction file in Drive on the test set: strict and loose F1, 95 % intervals over postings, a paired
bootstrap against the best system and the spread across seeds. No GPU needed.
"""),
        code(f"""
!pip -q install "yetkin @ git+https://github.com/{REPO}.git" matplotlib
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/yetkin"
"""),
        code("""
# the no-learning floor: a dictionary of every span seen in training
from yetkin.baseline import Lexicon
from yetkin.data import load
from yetkin.io import save_predictions
train, test = load("train"), load("test")
lex = Lexicon(train)
save_predictions(f"{OUT}/predictions/lexicon.jsonl", test, [lex.predict(s.tokens) for s in test],
                 meta={"name": "Lexicon (no learning)", "system": "Lexicon (no learning)", "entries": len(lex.entries)})
"""),
        code("""
from pathlib import Path
from IPython.display import Image, Markdown, display
from yetkin.compare import compare
files = sorted(p for p in Path(f"{OUT}/predictions").glob("*.jsonl") if not p.name.startswith("."))
compare(files, out=Path(f"{OUT}/results"))
display(Markdown(open(f"{OUT}/results/RESULTS.md").read()))
display(Image(f"{OUT}/results/f1.png"))
"""),
        code("""
# everything in one file to bring back to the repository
!cd "$OUT" && zip -qr /content/yetkin-results.zip predictions results models/*/run.json models/*/train_info.json 2>/dev/null; ls -la /content/yetkin-results.zip
from google.colab import files
files.download("/content/yetkin-results.zip")
"""),
    ]
)


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in (
        ("01_encoder", ENCODER),
        ("02_qlora", QLORA),
        ("03_compare", COMPARE),
    ):
        (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("wrote", f"notebooks/{name}.ipynb")


if __name__ == "__main__":
    main()
