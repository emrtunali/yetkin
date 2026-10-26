"""A transformer encoder with two token-classification heads, one per span type.

Skill and knowledge spans can overlap, so instead of one merged tag set the model has a separate
BIO head for each layer on a shared encoder, trained on the sum of both losses. Words are labelled
on their first sub-token; the other sub-tokens are ignored in the loss and in decoding.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .data import TYPES, Sentence, Span, bio_to_spans, spans_to_bio

TAGS = ["O", "B", "I"]
T2I = {t: i for i, t in enumerate(TAGS)}


class TwoHeadTagger(nn.Module):
    def __init__(self, name: str, dropout: float = 0.1):
        super().__init__()
        self.name = name
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(h, len(TAGS)) for t in TYPES})

    def forward(self, input_ids, attention_mask, labels: dict[str, torch.Tensor] | None = None):
        hidden = self.drop(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        logits = {t: head(hidden) for t, head in self.heads.items()}
        loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            loss = sum(ce(logits[t].reshape(-1, len(TAGS)), labels[t].reshape(-1)) for t in TYPES)
        return loss, logits

    def save(self, path: str | Path, tokenizer) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "encoder")
        torch.save(self.heads.state_dict(), path / "heads.pt")
        (path / "yetkin.json").write_text(json.dumps({"base": self.name, "tags": TAGS, "types": list(TYPES)}))

    @classmethod
    def load(cls, path: str | Path) -> tuple[TwoHeadTagger, object]:
        path = Path(path)
        model = cls(str(path / "encoder"))
        model.heads.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"))
        return model, AutoTokenizer.from_pretrained(path / "encoder")


def encode(sentences: list[Sentence], tok, max_length: int = 256) -> list[dict]:
    """Tokenise pre-split words; label only the first sub-token of each word."""
    out = []
    for s in sentences:
        enc = tok(s.tokens, is_split_into_words=True, truncation=True, max_length=max_length)
        words = enc.word_ids()
        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "words": words}
        for t in TYPES:
            tags = spans_to_bio(s.spans, len(s.tokens), t)
            lab, prev = [], None
            for w in words:
                lab.append(-100 if w is None or w == prev else T2I[tags[w]])
                prev = w
            item[t] = lab
        out.append(item)
    return out


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]  # noqa: E731
    return {
        "input_ids": torch.tensor(pad([b["input_ids"] for b in batch], pad_id)),
        "attention_mask": torch.tensor(pad([b["attention_mask"] for b in batch], 0)),
        "labels": {t: torch.tensor(pad([b[t] for b in batch], -100)) for t in TYPES},
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    name: str,
    train_set: list[Sentence],
    dev_set: list[Sentence],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-5,
    seed: int = 42,
    max_length: int = 256,
    log=print,
    on_epoch=None,
) -> tuple[TwoHeadTagger, object, dict]:
    """Fine-tune, keep the epoch with the best dev strict F1, return (model, tokenizer, history)."""
    from .metrics import score

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = (
        AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        if "roberta" in name.lower()
        else AutoTokenizer.from_pretrained(name)
    )
    model = TwoHeadTagger(name).to(device)
    data = encode(train_set, tok, max_length)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(data) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        random.shuffle(data)
        total = 0.0
        for i in range(0, len(data), batch_size):
            b = collate(data[i : i + batch_size], tok.pad_token_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss, _ = model(
                    b["input_ids"].to(device),
                    b["attention_mask"].to(device),
                    {t: v.to(device) for t, v in b["labels"].items()},
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        pred = predict(model, tok, dev_set, max_length=max_length)
        s = score([x.spans for x in dev_set], pred)
        f1 = s["strict_micro"]["f1"]
        history.append(
            {"epoch": epoch + 1, "loss": total / max(1, len(data) // batch_size), "dev_f1": f1, "dev": s}
        )
        log(f"epoch {epoch + 1}: loss {history[-1]['loss']:.4f}  dev strict F1 {f1:.4f}")
        if on_epoch:
            on_epoch(history[-1])
        if f1 > best:
            best = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, {"history": history, "best_dev_f1": best, "seed": seed, "base": name}


@torch.no_grad()
def predict(
    model: TwoHeadTagger, tok, sentences: list[Sentence], batch_size: int = 64, max_length: int = 256
) -> list[list[Span]]:
    model.eval()
    device = next(model.parameters()).device
    data = encode(sentences, tok, max_length)
    out = []
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        b = collate(chunk, tok.pad_token_id)
        _, logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        best = {t: logits[t].argmax(-1).cpu().tolist() for t in TYPES}
        for j, item in enumerate(chunk):
            n = len(sentences[i + j].tokens)
            spans = []
            for t in TYPES:
                tags, prev = ["O"] * n, None
                for k, w in enumerate(item["words"]):
                    if w is not None and w != prev:
                        tags[w] = TAGS[best[t][j][k]]
                    prev = w
                spans += bio_to_spans(tags, t)
            out.append(spans)
    return out
