# Yetkin

[![ci](https://github.com/emrtunali/yetkin/actions/workflows/ci.yml/badge.svg)](https://github.com/emrtunali/yetkin/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Which skills does a job posting ask for? Three ways to extract them, measured on the same test set.**

*Yetkin* is Turkish for *competent*. Given a job posting, the task is to mark every **skill** (something a person
does: *write clean code*, *manage stakeholders*) and every piece of **knowledge** (*Python*, *AWS*, *accounting*)
at the level of exact token spans. The question this repository answers:

> For skill extraction, should you fine-tune a small encoder, fine-tune a small LLM with QLoRA, or just prompt a
> bigger LLM? And what does each cost?

| | Approach | What is trained | Notebook |
|---|---|---|---|
| 1 | **Fine-tuned encoder**, two BIO heads (skill, knowledge) on a shared body. JobBERT and DeBERTa-v3, 3 seeds each | the whole encoder (110–185 M parameters) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emrtunali/yetkin/blob/main/notebooks/01_encoder.ipynb) |
| 2 | **QLoRA**: Qwen2.5-1.5B-Instruct in 4-bit, LoRA adapters on every linear layer, answering in JSON | about 1 % of the parameters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emrtunali/yetkin/blob/main/notebooks/02_qlora.ipynb) |
| 3 | **Prompting**: Qwen2.5-7B-Instruct in 4-bit, zero-shot and 5-shot | nothing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emrtunali/yetkin/blob/main/notebooks/03_prompting.ipynb) |
| – | **Lexicon**: every span seen in training, matched longest first | nothing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emrtunali/yetkin/blob/main/notebooks/04_compare.ipynb) |

All training runs on a free Colab T4. A demo for the best model runs on Hugging Face Spaces ([`space/`](space)).

## Results

SkillSpan test set: 3,569 sentences from 65 postings. Strict span F1 (exact boundaries), with 95 % bootstrap
intervals over postings. Full tables, loose F1, paired significance tests and the spread across seeds are written
to [`results/RESULTS.md`](results/RESULTS.md) by the comparison notebook.

| System | Strict F1 | 95 % interval | Status |
|---|---:|---|---|
| Lexicon (no learning) | 18.6 | 15.9–21.7 | measured |
| JobBERT, DeBERTa-v3 (3 seeds each) | – | – | run notebook 1 |
| QLoRA · Qwen2.5-1.5B | – | – | run notebook 2 |
| Prompted · Qwen2.5-7B, 0- and 5-shot | – | – | run notebook 3 |

The lexicon is the floor: it knows every skill name in the training postings and still finds fewer than one
span in five exactly, because the same words are a skill in one sentence and not in the next.

## How it is evaluated

- **Data.** [SkillSpan](https://huggingface.co/datasets/jjzha/skillspan) (Zhang et al., NAACL 2022; CC BY 4.0):
  263 English job postings annotated token by token with two independent layers, skill and knowledge. The
  layers overlap in about 1 % of tokens, so they are kept as two layers instead of being merged into one tag set.
- **One scorer for every system.** Generated answers are JSON lists of strings. They are aligned back onto the
  sentence's tokens and scored exactly like a tagger's BIO output. A string that does not occur in the sentence
  is dropped, so a hallucinated skill can only cost precision.
- **Strict and loose F1.** Strict needs exact boundaries (the paper's metric). Loose accepts any overlap and shows
  how much of the error is boundary disagreement rather than a missed skill.
- **Uncertainty.** Sentences of one posting are not independent, so intervals resample whole postings. A paired
  bootstrap tests each system against the best one. Encoders are trained with three seeds; the model selected
  for the demo is chosen on the dev set, never on test.
- **Cost.** Every run records training minutes, trainable parameters, peak GPU memory and seconds per 1,000
  test sentences.

## Tools

PyTorch · Hugging Face Transformers · PEFT (LoRA, QLoRA) · bitsandbytes (4-bit NF4) · Weights & Biases (optional)
· Hugging Face Hub · Gradio on Spaces · pytest and GitHub Actions.

## Running it

**In Colab.** Open a notebook from the table above, choose *Runtime → Change runtime type → T4 GPU*, then *Run all*.
Results are written to Google Drive, so a disconnected session continues where it stopped. Two optional secrets
(the 🔑 panel): `HF_TOKEN` to publish the trained models, `WANDB_API_KEY` to log the runs.

**Locally**, for the code and the tests (no GPU needed):

```bash
pip install -e ".[dev]"
pytest -m "not slow"                       # data, metrics, alignment, notebooks compile
pip install -e ".[train]" && pytest -m slow  # trains tiny models on CPU through the same code paths
python -m yetkin.baseline                  # the lexicon floor
python -m yetkin.compare results/predictions/*.jsonl
```

The notebooks are generated from [`scripts/make_notebooks.py`](scripts/make_notebooks.py); every code path they
use lives in the package and is tested.

```
src/yetkin/
  data.py        SkillSpan loading, BIO <-> spans, long-sentence chunking
  encoder.py     two-head token classifier, training loop, prediction
  llm.py         QLoRA fine-tuning and batched generation
  generative.py  prompts, JSON parsing, aligning answers to tokens
  metrics.py     strict/loose span F1, bootstrap intervals, paired tests
  baseline.py    lexicon baseline
  compare.py     results table and chart
  posting.py     raw posting -> sentences; skills report and gap for the demo
```

## References

- Zhang, Jensen, Sonniks, Plank (2022). *SkillSpan: Hard and Soft Skill Extraction from English Job Postings.* NAACL.
- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- He, Gao, Chen (2023). *DeBERTaV3.* ICLR.

## License

Code: [MIT](LICENSE). SkillSpan data: CC BY 4.0, by its authors.
