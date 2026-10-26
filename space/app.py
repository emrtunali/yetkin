"""Hugging Face Space: paste a job posting, see its skills and knowledge, compare with your own.

Runs the fine-tuned two-head encoder on CPU. Set MODEL_REPO in the Space settings to the model
repository the first notebook published (default below).
"""

import os

import gradio as gr
from huggingface_hub import snapshot_download

from yetkin.encoder import TwoHeadTagger, predict
from yetkin.posting import gap, report, sentences

MODEL_REPO = os.environ.get("MODEL_REPO", "emrtunali/yetkin-skill-tagger")
model, tok = TwoHeadTagger.load(snapshot_download(MODEL_REPO))

EXAMPLE = """Machine Learning Engineer
You will design, train and deploy machine learning models for our recommendation platform.
Requirements:
- 3+ years of experience with Python and PyTorch
- Experience with MLOps tools such as MLflow, Docker and Kubernetes
- Strong communication skills and the ability to work in a cross-functional team
- Familiarity with AWS or GCP; knowledge of SQL
- You enjoy mentoring junior engineers and writing clean, tested code"""


def run(text: str, mine: str):
    sents = sentences(text)
    if not sents:
        return [], "", ""
    pred = predict(model, tok, sents)
    highlighted = []
    for s, spans in zip(sents, pred, strict=True):
        label = ["" for _ in s.tokens]
        for sp in spans:
            for i in range(sp.start, sp.end):
                label[i] = "Knowledge" if sp.type == "knowledge" else "Skill"
        for tok_, lab in zip(s.tokens, label, strict=True):
            highlighted.append((tok_ + " ", lab or None))
        highlighted.append(("\n", None))
    found = report(sents, pred)
    table = "\n".join(
        [
            f"**Knowledge** ({len(found['knowledge'])}): " + ", ".join(t for t, _ in found["knowledge"]),
            "",
            f"**Skills** ({len(found['skill'])}): " + ", ".join(t for t, _ in found["skill"]),
        ]
    )
    if mine.strip():
        g = gap(found, mine)
        fit = (
            f"**You have** ({len(g['match'])}): "
            + (", ".join(g["match"]) or "—")
            + f"\n\n**Missing** ({len(g['missing'])}): "
            + (", ".join(g["missing"]) or "—")
        )
    else:
        fit = "List your skills on the left to see what matches and what is missing."
    return highlighted, table, fit


with gr.Blocks(title="Yetkin · skills in a job posting") as demo:
    gr.Markdown(
        "# Yetkin\nPaste a job posting. A fine-tuned encoder marks the **skills** (things you do) "
        "and **knowledge** (tools, fields) it asks for."
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Job posting", value=EXAMPLE, lines=14)
            mine = gr.Textbox(
                label="Your skills (comma separated, optional)",
                placeholder="Python, SQL, Docker, communication",
            )
            go = gr.Button("Extract", variant="primary")
        with gr.Column():
            out = gr.HighlightedText(
                label="Marked posting",
                color_map={"Skill": "#eb6834", "Knowledge": "#2a78d6"},
                combine_adjacent=True,
            )
            table = gr.Markdown()
            fit = gr.Markdown()
    go.click(run, [text, mine], [out, table, fit])
    demo.load(run, [text, mine], [out, table, fit])

if __name__ == "__main__":
    demo.launch()
