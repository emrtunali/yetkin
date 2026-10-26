"""Instruction-tuned LLMs for span extraction: prompting, and QLoRA fine-tuning.

QLoRA (Dettmers et al., 2023): the base model is loaded in 4-bit NF4, frozen, and small low-rank
adapters on every linear layer are trained in 16-bit. Only the answer tokens contribute to the
loss, so the model learns to answer rather than to repeat the instructions.
"""

from __future__ import annotations

import math
import random

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import Sentence, Span
from .generative import align, messages, parse_answer, target

_V5 = int(transformers.__version__.split(".")[0]) >= 5


def load(name: str, four_bit: bool = True, adapter: str | None = None):
    """Model and tokenizer; 4-bit when a GPU is present and ``four_bit`` is set."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # generation pads on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if four_bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kw["device_map"] = "auto"
    elif torch.cuda.is_available():
        # transformers 5 renamed torch_dtype to dtype
        kw.update({"dtype" if _V5 else "torch_dtype": torch.float16, "device_map": "auto"})
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def _prompt(tok, s: Sentence, shots) -> str:
    return tok.apply_chat_template(messages(s, shots), tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def extract(
    model,
    tok,
    sentences: list[Sentence],
    shots: list[Sentence] = (),
    batch_size: int = 16,
    max_new_tokens: int = 160,
    log=None,
) -> tuple[list[list[Span]], list[str]]:
    """Greedy generation for every sentence -> (spans, raw answers)."""
    model.eval()
    device = next(model.parameters()).device
    spans, raw = [], []
    # sort by length so batches pad little; restore the order at the end
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i].tokens))
    res: dict[int, tuple[list[Span], str]] = {}
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok(
            [_prompt(tok, sentences[i], shots) for i in idx],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, t in zip(idx, texts, strict=True):
            res[i] = (align(sentences[i].tokens, parse_answer(t)), t)
        if log and (b // batch_size) % 20 == 0:
            log(f"  {min(b + batch_size, len(order))}/{len(order)}")
    for i in range(len(sentences)):
        spans.append(res[i][0])
        raw.append(res[i][1])
    return spans, raw


def sft_example(tok, s: Sentence, max_length: int = 512) -> dict:
    """Token ids for prompt + answer, with the prompt masked out of the labels."""
    prompt = tok.apply_chat_template(messages(s), tokenize=False, add_generation_prompt=True)
    answer = target(s) + (tok.eos_token or "")
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_length]
    labels = ([-100] * len(p) + a)[:max_length]
    return {"input_ids": ids, "labels": labels}


def qlora(
    name: str,
    train_set: list[Sentence],
    out_dir: str,
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 2e-4,
    r: int = 16,
    alpha: int = 32,
    four_bit: bool = True,
    seed: int = 42,
    max_length: int = 512,
    log=print,
    on_step=None,
) -> dict:
    """Train LoRA adapters on (4-bit) ``name``; saves the adapter to ``out_dir``."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(seed)
    torch.manual_seed(seed)
    model, tok = load(name, four_bit=four_bit)
    tok.padding_side = "right"  # training pads on the right
    if four_bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f} %)")
    data = [sft_example(tok, s, max_length) for s in train_set]
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(data) / (batch_size * grad_accum))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    history, step = [], 0
    model.train()
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            n = max(len(x["input_ids"]) for x in batch)
            ids = torch.tensor(
                [x["input_ids"] + [tok.pad_token_id] * (n - len(x["input_ids"])) for x in batch],
                device=device,
            )
            lab = torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in batch], device=device)
            att = (
                torch.arange(n, device=device)[None, :]
                < torch.tensor([len(x["input_ids"]) for x in batch], device=device)[:, None]
            ).long()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / grad_accum
            loss.backward()
            if (i // batch_size + 1) % grad_accum == 0 or i + batch_size >= len(data):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                rec = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": loss.item() * grad_accum,
                    "lr": sched.get_last_lr()[0],
                }
                history.append(rec)
                if on_step:
                    on_step(rec)
                if step % 20 == 0:
                    log(f"step {step}/{steps}  loss {rec['loss']:.4f}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {
        "history": history,
        "trainable": trainable,
        "total": total,
        "base": name,
        "r": r,
        "alpha": alpha,
        "epochs": epochs,
    }
