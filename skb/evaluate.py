"""Evaluate every model of the ladder on the SAME held-out test set (WikiText-2 test).

Metrics
* word-level perplexity  PPL = exp( - sum log p(test text) / #words )
  Each model scores the exact same normalized test paragraphs with its own
  byte-level tokenizer, and the log-probability is divided by the same number
  of whitespace words, so the numbers are directly comparable.
* top-3 next-word accuracy on 1,000 fixed test positions (seed 42): is the true
  next word among the model's three suggestions?
* latency per suggestion (ms): wall-clock of one `next_words` call, averaged
  over all 1,000 calls (>=100 required), CPU only, after 5 warm-up calls.
* model size (MB) on disk.
"""
import json
import math
import os
import platform
import random
import re
import subprocess
import time

import numpy as np

from . import config
from .lm import next_words

TARGET_RE = re.compile(r"^[A-Za-z]+$")


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    info = {"cpu": cpu, "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["quantized_engine"] = torch.backends.quantized.engine
    except Exception:
        pass
    return info


def sample_positions(paragraphs, n, seed=config.SEED, max_ctx_words=40):
    cands = []
    for pi, p in enumerate(paragraphs):
        words = p.split(" ")
        for j in range(3, len(words)):
            if TARGET_RE.match(words[j]):
                cands.append((pi, j))
    rng = random.Random(seed)
    picked = rng.sample(cands, min(n, len(cands)))
    out = []
    for pi, j in picked:
        words = paragraphs[pi].split(" ")
        out.append({"context": " ".join(words[max(0, j - max_ctx_words):j]), "target": words[j]})
    return out


def perplexity(lm, paragraphs):
    t0 = time.time()
    total_lp = sum(lm.paragraph_logprob(p) for p in paragraphs)
    n_words = sum(len(p.split()) for p in paragraphs)
    return math.exp(-total_lp / n_words), n_words, time.time() - t0


def top3_and_latency(lm, positions, warmup=5):
    for pos in positions[:warmup]:
        next_words(lm, pos["context"])
    hits, times, preds = 0, [], []
    for pos in positions:
        t0 = time.perf_counter()
        sugg = next_words(lm, pos["context"])
        times.append((time.perf_counter() - t0) * 1000)
        words = [w for w, _ in sugg]
        hit = pos["target"].lower() in [w.lower() for w in words]
        hits += hit
        preds.append(words)
    t = np.asarray(times)
    return {
        "top3_acc": hits / len(positions),
        "latency_ms_mean": float(t.mean()),
        "latency_ms_p50": float(np.percentile(t, 50)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "latency_runs": len(t),
    }, preds


DEMO_CONTEXTS = [
    "The film was released in the United",
    "He was born in",
    "The album debuted at number one on the",
    "During the Second World",
    "The species is found in",
]


def run(models, test_paragraphs, n_positions=config.EVAL_TOP3_SAMPLES, log=print):
    positions = sample_positions(test_paragraphs, n_positions)
    results = {"hardware": hardware(), "test_paragraphs": len(test_paragraphs),
               "top3_positions": len(positions), "models": [], "examples": [], "failures": []}
    all_preds = {}
    for key, lm in models:
        log(f"[eval] {lm.name}: perplexity ...")
        ppl, n_words, ppl_sec = perplexity(lm, test_paragraphs)
        log(f"[eval] {lm.name}: PPL {ppl:.2f} over {n_words:,} words ({ppl_sec:.0f}s); top-3 + latency ...")
        m, preds = top3_and_latency(lm, positions)
        all_preds[key] = preds
        row = {"key": key, "model": lm.name, "precision": lm.precision, "word_ppl": ppl,
               "test_words": n_words, "size_mb": lm.size_mb(), **m}
        if hasattr(lm, "use_cache"):
            row["kv_cache"] = lm.use_cache
            row["kv_cache_check_maxdiff"] = lm.cache_check
        results["models"].append(row)
        log(f"[eval] {lm.name}: top-3 {m['top3_acc']:.3f}, {m['latency_ms_mean']:.1f} ms, {row['size_mb']:.1f} MB")
        results["examples"].append({"model": lm.name,
                                    "suggestions": {c: [w for w, _ in next_words(lm, c)] for c in DEMO_CONTEXTS}})

    # failure analysis for the deployed model (last in the ladder)
    key = models[-1][0]
    for pos, words in zip(positions, all_preds[key]):
        if pos["target"].lower() not in [w.lower() for w in words]:
            results["failures"].append({"context": pos["context"][-120:], "target": pos["target"], "predicted": words})
        if len(results["failures"]) >= 15:
            break
    # agreement between FP32 and INT8 GPT-2 suggestions
    if "gpt2" in all_preds and "gpt2_int8" in all_preds:
        same = [a[:1] == b[:1] for a, b in zip(all_preds["gpt2"], all_preds["gpt2_int8"])]
        results["fp32_int8_top1_agreement"] = float(np.mean(same))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results)
    plot(results)
    return results


def write_markdown(res):
    hw = res["hardware"]
    lines = ["| Model | Precision | Word PPL ↓ | Top-3 acc ↑ | Size (MB) | Latency (ms, mean) |",
             "|---|---|---|---|---|---|"]
    for r in res["models"]:
        lines.append(f"| {r['model']} | {r['precision']} | {r['word_ppl']:.1f} | {100 * r['top3_acc']:.1f}% | "
                     f"{r['size_mb']:.1f} | {r['latency_ms_mean']:.1f} |")
    lines.append("")
    lines.append(f"Test set: WikiText-2 test, {res['models'][0]['test_words']:,} words; top-3 accuracy on "
                 f"{res['top3_positions']} positions; latency averaged over {res['models'][0]['latency_runs']} runs "
                 f"per model on {hw.get('cpu')} CPU ({hw.get('torch_threads', '?')} threads), {hw.get('os')}.")
    (config.RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = res["models"]
    names = [r["model"].replace(" fine-tuned", "\nfine-tuned").replace(" (", "\n(") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title, log_scale in [(axes[0], "word_ppl", "Word perplexity (lower is better)", True),
                                      (axes[1], "top3_acc", "Top-3 next-word accuracy", False),
                                      (axes[2], "latency_ms_mean", "Latency per suggestion (ms, CPU)", False)]:
        vals = [r[key] * (100 if key == "top3_acc" else 1) for r in rows]
        bars = ax.bar(range(len(rows)), vals, color="#4C72B0")
        bars[-1].set_color("#DD8452")
        ax.set_xticks(range(len(rows)), names, fontsize=7)
        ax.set_title(title, fontsize=10)
        if log_scale:
            ax.set_yscale("log")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}{'%' if key == 'top3_acc' else ''}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "results_chart.png", dpi=200)
