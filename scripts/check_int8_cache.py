"""Decide with data whether the INT8 model may use the KV cache.

Dynamic INT8 quantization computes one activation scale per call. Without the cache the whole
sequence (including the first token, whose GPT-2 activations are known to be extremely large) is
quantized together; with the cache only the few new tokens are. The two paths therefore give
slightly different results, and the question is which one stays closer to the FP32 model.

    python scripts/check_int8_cache.py      # ~5 min, writes results/int8_cache_check.json

On 200 fixed test positions (seed 42) it compares, for INT8 without and with the cache:
top-1 agreement with FP32, top-3 accuracy against the true next word, and latency.
The cache is enabled for INT8 only if it agrees with FP32 at least as often as the
no-cache path; GPT2LM reads the decision from the JSON file.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from skb import config, data, evaluate  # noqa: E402
from skb.gpt2 import GPT2LM, INT8_CACHE_FILE  # noqa: E402
from skb.lm import next_words  # noqa: E402


def run_model(lm, positions):
    preds, times = [], []
    next_words(lm, positions[0]["context"])                    # warm-up
    for p in positions:
        t0 = time.perf_counter()
        preds.append([w for w, _ in next_words(lm, p["context"])])
        times.append((time.perf_counter() - t0) * 1000)
    return preds, float(np.mean(times))


def main(n=200):
    positions = evaluate.sample_positions(data.load_split("test"), n)
    fp32 = GPT2LM(int8=False)
    int8 = GPT2LM(int8=True)
    ref, ref_ms = run_model(fp32, positions)
    out = {"positions": n, "fp32_ms": ref_ms,
           "fp32_top3_acc": float(np.mean([p["target"].lower() in [w.lower() for w in r] for p, r in zip(positions, ref)]))}
    for name, use in (("int8_no_cache", False), ("int8_cache", True)):
        int8.use_cache, int8._cache = use, None
        preds, ms = run_model(int8, positions)
        out[name] = {
            "top1_agreement_with_fp32": float(np.mean([a[:1] == b[:1] for a, b in zip(preds, ref)])),
            "top3_acc": float(np.mean([p["target"].lower() in [w.lower() for w in r] for p, r in zip(positions, preds)])),
            "latency_ms": ms,
        }
        print(name, out[name])
    a, b = out["int8_cache"], out["int8_no_cache"]
    out["cache_allowed"] = bool(a["top1_agreement_with_fp32"] >= b["top1_agreement_with_fp32"])
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INT8_CACHE_FILE.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nINT8 KV cache", "ALLOWED" if out["cache_allowed"] else "NOT allowed",
          "-> now run:  python train_all.py")


if __name__ == "__main__":
    main()
