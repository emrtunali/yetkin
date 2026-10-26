"""The training code paths, end to end with tiny models on CPU (a couple of minutes, downloads ~300 MB).

    pytest -m slow

They check that the same code the Colab notebooks run trains, saves, reloads and predicts, not
that tiny models are any good.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from yetkin.data import Span, parse  # noqa: E402
from yetkin.metrics import score  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "sample.jsonl"
SENTS = [parse(json.loads(line)) for line in FIX.read_text().splitlines()]


@pytest.mark.slow
def test_two_head_encoder_trains_saves_and_predicts(tmp_path):
    from yetkin import encoder

    model, tok, info = encoder.train(
        "hf-internal-testing/tiny-random-BertModel",
        SENTS,
        SENTS[:10],
        epochs=2,
        batch_size=8,
        lr=5e-4,
        log=lambda *_: None,
    )
    assert len(info["history"]) == 2 and info["history"][-1]["loss"] > 0
    pred = encoder.predict(model, tok, SENTS)
    assert len(pred) == len(SENTS) and all(isinstance(s, Span) for p in pred for s in p)
    model.save(tmp_path / "m", tok)
    again, tok2 = encoder.TwoHeadTagger.load(tmp_path / "m")
    assert encoder.predict(again, tok2, SENTS) == pred


@pytest.mark.slow
def test_fp16_checkpoint_trains_in_fp32(tmp_path):
    """DeBERTa-v3 ships fp16 weights; GradScaler refuses to unscale fp16 gradients."""
    from transformers import AutoModel, AutoTokenizer

    from yetkin import encoder

    name = "hf-internal-testing/tiny-random-BertModel"
    AutoModel.from_pretrained(name).half().save_pretrained(tmp_path / "half")
    AutoTokenizer.from_pretrained(name).save_pretrained(tmp_path / "half")
    model = encoder.TwoHeadTagger(str(tmp_path / "half"))
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.slow
def test_lora_trains_and_extracts(tmp_path):
    from yetkin import llm

    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    info = llm.qlora(
        name,
        SENTS[:16],
        str(tmp_path / "adapter"),
        epochs=1,
        batch_size=4,
        grad_accum=1,
        four_bit=False,
        log=lambda *_: None,
    )
    assert info["trainable"] < info["total"] * 0.1 and info["history"]
    model, tok = llm.load(name, four_bit=False, adapter=str(tmp_path / "adapter"))
    pred, raw = llm.extract(model, tok, SENTS[:6], batch_size=3, max_new_tokens=24)
    assert len(pred) == 6 and len(raw) == 6
    score([s.spans for s in SENTS[:6]], pred)  # whatever it says, it must be scoreable


@pytest.mark.slow
def test_lora_resumes_from_a_checkpoint(tmp_path):
    from yetkin import llm

    class Disconnect(Exception):
        pass

    def stop_at_3(rec):
        if rec["step"] == 3:
            raise Disconnect

    kw = dict(epochs=2, batch_size=4, grad_accum=1, four_bit=False, checkpoint_dir=str(tmp_path / "ckpt"))
    kw["save_every"] = 2
    name, data, out = "HuggingFaceTB/SmolLM2-135M-Instruct", SENTS[:12], str(tmp_path / "adapter")
    with pytest.raises(Disconnect):
        llm.qlora(name, data, out, on_step=stop_at_3, log=lambda *_: None, **kw)
    assert (tmp_path / "ckpt" / "state.pt").exists()
    lines = []
    info = llm.qlora(name, data, out, log=lines.append, **kw)
    assert "resuming at step 2/6" in lines
    assert [h["step"] for h in info["history"]] == [1, 2, 3, 4, 5, 6]
    assert not (tmp_path / "ckpt").exists() and (tmp_path / "adapter" / "adapter_config.json").exists()


@pytest.mark.slow
def test_prompt_masks_everything_but_the_answer():
    from transformers import AutoTokenizer

    from yetkin.generative import target
    from yetkin.llm import sft_example

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    ex = sft_example(tok, SENTS[0])
    answer = [i for i, lab in zip(ex["input_ids"], ex["labels"], strict=True) if lab != -100]
    assert target(SENTS[0]) in tok.decode(answer)
    assert ex["labels"][0] == -100
