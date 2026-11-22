"""Reproduce every number in the presentation:

    python train_all.py           # full run (~30-40 min on an Apple M2)
    python train_all.py --quick   # 3-minute smoke test on a tiny subset (separate folders)

Steps (each one is skipped if its output already exists; use --force to redo):
  1. download + normalize WikiText-2            -> data/
  2. train byte-level BPE (8k)                  -> artifacts/bpe/
  3. count trigram models (add-one, Kneser-Ney) -> artifacts/ngram/
  4. train feed-forward NN LM with SGD          -> artifacts/ffnn/
  5. fine-tune GPT-2 small                      -> artifacts/gpt2-finetuned/
  6. dynamic INT8 quantization of GPT-2         -> artifacts/gpt2-int8/
  7. evaluate all models on the same test set   -> results/
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="tiny smoke test (writes to *_quick folders)")
ap.add_argument("--force", action="store_true", help="redo every step")
ap.add_argument("--skip-eval", action="store_true")
ap.add_argument("--requantize", action="store_true", help="redo only the INT8 quantization (then evaluate)")
args = ap.parse_args()
if args.quick:
    os.environ["SKB_QUICK"] = "1"

from skb import config, data  # noqa: E402  (config reads SKB_QUICK)


def main():
    t_start = time.time()
    data.prepare()
    train = data.load_split("train")
    valid = data.load_split("valid")
    test = data.load_split("test")
    if args.quick:
        train, valid, test = train[:1500], valid[:100], test[:60]

    from skb import bpe
    if args.force or not (bpe.BPE_DIR / "merges.txt").exists():
        tok = bpe.train()
    else:
        tok = bpe.BPETokenizer()

    from skb.models import ADDONE_FILE, KN_FILE
    from skb.ngram import AddOneNGram, KneserNeyNGram
    if args.force or not (ADDONE_FILE.exists() and KN_FILE.exists()):
        print("[ngram] encoding training text ...")
        train_ids = tok.encode_batch(train)
        t0 = time.time()
        AddOneNGram(tok).fit(train_ids).save(ADDONE_FILE)
        KneserNeyNGram(tok).fit(train_ids).save(KN_FILE)
        print(f"[ngram] done ({time.time() - t0:.0f}s)")

    from skb import ffnn
    if args.force or not (ffnn.FFNN_DIR / "ffnn.pt").exists():
        ffnn.train(tok, tok.encode_batch(train), tok.encode_batch(valid),
                   epochs=1 if args.quick else config.FFNN_EPOCHS)

    from skb import gpt2
    if args.force or not (gpt2.FT_DIR / "config.json").exists():
        gpt2.finetune(train, valid, max_steps=20 if args.quick else None)
    if args.force or args.requantize or not gpt2.Q_FILE.exists():
        gpt2.quantize()

    if not args.skip_eval:
        from skb import evaluate
        from skb.models import load_ladder
        evaluate.run(load_ladder(), test, n_positions=60 if args.quick else config.EVAL_TOP3_SAMPLES)
    print(f"\nAll done in {(time.time() - t_start) / 60:.1f} min. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
