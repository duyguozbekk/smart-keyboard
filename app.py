"""Smart Keyboard - Gradio application.

    python app.py            # opens http://127.0.0.1:7860

As you type, every model of the ladder suggests the next three words, side by
side, with its response time. The keyboard itself is driven by the deployed
model, the INT8-quantized fine-tuned GPT-2 (top row). Clicking any suggestion
inserts it into the text box.
"""
import argparse
import sys

import gradio as gr

from skb import config
from skb.lm import timed_next_words

REQUIRED = [
    "bpe/merges.txt", "bpe/vocab.json",
    "ngram/trigram_addone.pkl", "ngram/trigram_kn.pkl",
    "ffnn/ffnn.pt",
    "gpt2-int8/gpt2_int8_state_dict.pt", "gpt2-int8/config.json",
]


def ensure_weights(need_fp32):
    files = REQUIRED + (["gpt2-finetuned/config.json"] if need_fp32 else [])
    missing = [f for f in files if not (config.ART_DIR / f).exists()]
    if not missing:
        return
    print(f"[app] {len(missing)} weight files missing -> downloading from huggingface.co/{config.WEIGHTS_REPO}")
    try:
        from huggingface_hub import snapshot_download
        patterns = None if need_fp32 else ["bpe/*", "ngram/*", "ffnn/*", "gpt2-int8/*"]
        snapshot_download(repo_id=config.WEIGHTS_REPO, local_dir=str(config.ART_DIR), allow_patterns=patterns)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[app] could not download weights ({e}).\nTrain them locally instead:  python train_all.py")


def build(models):
    deployed = models[-1]
    order = [deployed] + models[:-1]           # deployed model first

    with gr.Blocks(title="Smart Keyboard - next-word prediction") as demo:
        gr.Markdown(
            "# ⌨️ Smart Keyboard — next-word prediction\n"
            "Domain: English Wikipedia articles (WikiText-2). Type a sentence; each model suggests the next "
            "**3 words** with its response time (CPU). The keyboard runs the **INT8-quantized GPT-2** (top row). "
            "Click a suggestion to insert it.")
        text = gr.Textbox(label="Your text", lines=4, autofocus=True,
                          placeholder="e.g. The album debuted at number one on the")
        rows = []
        for i, (key, lm) in enumerate(order):
            with gr.Row(equal_height=True):
                info = gr.Markdown()
                btns = [gr.Button("…", variant="primary" if i == 0 else "secondary", min_width=90)
                        for _ in range(config.TOPK_WORDS)]
            rows.append((info, btns))
        gr.Examples(examples=[["The film was released in the United"], ["He was born in"],
                              ["During the Second World"], ["The species is found in"]], inputs=text)

        def suggest(t):
            out = []
            for i, (key, lm) in enumerate(order):
                sugg, ms = timed_next_words(lm, t or "")
                tag = " — **used by the keyboard**" if i == 0 else ""
                out.append(f"**{lm.name}** · {lm.precision} · `{ms:.1f} ms`{tag}")
                words = [w for w, _ in sugg] + ["…"] * config.TOPK_WORDS
                out.extend(gr.update(value=w) for w in words[:config.TOPK_WORDS])
            return out

        def insert(t, word):
            if not word or word == "…":
                return t
            t = (t or "").rstrip()
            return (t + " " if t else "") + word + " "

        outputs = [c for info, btns in rows for c in [info, *btns]]
        text.change(suggest, inputs=text, outputs=outputs, trigger_mode="always_last", show_progress="hidden")
        demo.load(suggest, inputs=text, outputs=outputs)
        for _, btns in rows:
            for b in btns:
                b.click(insert, inputs=[text, b], outputs=text)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fp32", action="store_true", help="do not load the FP32 GPT-2 (saves ~0.5 GB RAM)")
    ap.add_argument("--share", action="store_true", help="also create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    ensure_weights(need_fp32=not args.no_fp32)
    from skb.models import load_ladder
    models = load_ladder(include_fp32_gpt2=not args.no_fp32)
    build(models).queue().launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
