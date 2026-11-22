# ⌨️ Smart Keyboard — Next-Word Prediction (SWE015 Homework, Use Case 5)

**Duygu Özbek · 2409011048 · SWE015 Introduction to Large Language Models**

An on-device-style keyboard for **English Wikipedia-style text** that suggests the next three words while
you type. Four models of increasing sophistication are trained and compared on the **same held-out test
set**, and the application runs the **INT8-quantized GPT-2**.

| Ladder step | Model | Technique area |
|---|---|---|
| Classical baseline | Trigram LM over our own BPE tokens, **add-one** and **Kneser-Ney** smoothing | B |
| Neural network | **Feed-forward neural LM** (embedding → hidden → softmax), trained from scratch with **SGD + backpropagation** | A |
| Transformer | **GPT-2 small** (124M) fine-tuned on the corpus | C |
| Quantized Transformer | GPT-2 fine-tuned, **dynamic INT8 quantization** (PyTorch) — used by the app | D |

## 1. Setup

Requires Python **3.10 – 3.12** (macOS / Linux / Windows; an Apple-Silicon or CUDA GPU speeds up training).

```bash
git clone https://github.com/duyguozbekk/smart-keyboard.git
cd smart-keyboard
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the application (single command)

```bash
python app.py
```

Opens <http://127.0.0.1:7860>. On the first start the trained weights are downloaded automatically from the
Hugging Face Hub (**[huggingface.co/duyguozbekk/smart-keyboard-weights](https://huggingface.co/duyguozbekk/smart-keyboard-weights)**);
weights are never stored in this git repository. Use `python app.py --no-fp32` on machines with little RAM.

## 3. Reproduce every number (optional)

```bash
python train_all.py --quick   # 3-minute smoke test on a tiny subset (writes to *_quick folders)
python train_all.py           # full pipeline, ~30–40 min on an Apple M2
```

`train_all.py` downloads and normalizes WikiText-2, trains the BPE tokenizer, counts both trigram models,
trains the feed-forward network, fine-tunes GPT-2, quantizes it, and evaluates all five models. It writes
`results/results.json`, `results/results.md` and `results/results_chart.png`. Each step is skipped when its
output already exists (`--force` redoes everything). Unit tests for the pure-Python parts: `python tests/test_core.py`.

To publish your own trained weights: `hf auth login`, then `python scripts/upload_weights.py`.

## 4. Data

* **WikiText-2 (raw)** — Merity et al., *Pointer Sentinel Mixture Models*, 2016. Loaded from
  [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) (`wikitext-2-raw-v1`), CC BY-SA 3.0.
  Official splits are kept: **train** for all training, **validation** for model selection, **test** for every
  reported number. Exact paragraph and word counts per split are written to `results/data_stats.json`.
* Preprocessing (`skb/data.py`, regular expressions): headings and empty lines removed, `@-@ @,@ @.@` escapes undone,
  spaces before punctuation / after brackets / before clitics removed, whitespace collapsed.
* **GPT-2 small** pretrained weights — Radford et al., 2019 ([`gpt2`](https://huggingface.co/gpt2), MIT license).

## 5. How it works

```
user types ─► Gradio text box ─► context (last ≤128 tokens)
                                   │
         ┌─────────────────────────┼───────────────────────────────┐
         ▼                         ▼                               ▼
  our byte-level BPE (8k)    our byte-level BPE (8k)        GPT-2 byte-level BPE (50k)
  trigram add-one / KN        feed-forward NN (FP32)        GPT-2 FP32  |  GPT-2 INT8 ◄ deployed
         └───────────── next-token log-probabilities ─────────────────┘
                                   ▼
          word beam search (width 6, ≤4 sub-word pieces per word)
                                   ▼
           top-3 words + latency per model ─► clickable buttons
```

* **Word assembly** (`skb/lm.py`): every model predicts sub-word tokens; a small beam search expands
  word-start tokens (`" Brit"`), then continuation tokens (`"ain"`), and closes a word with the probability
  of the next token being a boundary (space / punctuation). Identical words reached by different paths are summed.
* **Perplexity** is word-level: `exp(−Σ log p(test text) / #words)`. All tokenizers are lossless byte-level BPEs,
  so every model scores exactly the same characters and the numbers are comparable.
* **Top-3 accuracy**: 1,000 fixed test positions (seed 42, target = alphabetic word, context = up to 40 previous words).
* **Latency**: one full suggestion (`next_words`) on CPU, averaged over the same 1,000 calls after 5 warm-ups.
* **Quantization** (`skb/gpt2.py`): Hugging Face GPT-2 uses `Conv1D` layers, which are converted to equivalent
  `nn.Linear` layers; `torch.ao.quantization.quantize_dynamic` then stores the Linear weights of all 12
  transformer blocks (attention + MLP) in INT8. Embeddings, LayerNorms and the embedding-tied LM head stay FP32.
* **KV cache**: for GPT-2 the context is encoded once per suggestion and the word beam search only feeds the new
  sub-word tokens; at load time the cached path is checked against full recomputation.

## 6. Results

| Model | Precision | Word PPL ↓ | Top-3 acc ↑ | Size (MB) | Latency (ms, mean) |
|---|---|---|---|---|---|
| Trigram (add-one) | float64 counts | 415126.6 | 22.0% | 35.9 | 2.1 |
| Trigram (Kneser-Ney) | float64 counts | 2940.7 | 29.0% | 42.7 | 2.7 |
| Feed-forward NN LM | FP32 | 8316.3 | 26.5% | 20.8 | 8.2 |
| GPT-2 fine-tuned | FP32 | 89.2 | 48.8% | 474.7 | 155.4 |
| GPT-2 fine-tuned INT8 | INT8 dynamic (blocks), FP32 head | 108.3 | 47.0% | 231.8 | 166.1 |

Test set: WikiText-2 test, 201,560 words; top-3 accuracy on 1000 positions; latency averaged over 1000 runs per model on Apple M2 CPU (4 threads), Darwin 27.2.0.

Both GPT-2 models use the KV cache. For INT8, dynamic quantization makes the cached and uncached paths give
slightly different log-probabilities, so `scripts/check_int8_cache.py` checked on 200 test positions that the
cache keeps the same agreement with FP32 (80.5% top-1 in both cases, 48.0% top-3) while cutting latency from
603.7 to 139.4 ms; the decision is stored in `results/int8_cache_check.json` and read by the app.
Raw numbers: [`results/results.json`](results/results.json), chart: `results/results_chart.png`.

## 7. Repository layout

```
app.py               Gradio application (single run command)
train_all.py         full training + evaluation pipeline
skb/config.py        paths and hyper-parameters
skb/data.py          WikiText-2 download + regex normalization          (B)
skb/bpe.py           byte-level BPE tokenizer trained on the corpus     (B)
skb/ngram.py         trigram LMs: add-one and interpolated Kneser-Ney   (B)
skb/ffnn.py          feed-forward neural LM trained with SGD            (A)
skb/gpt2.py          GPT-2 fine-tuning (C) and INT8 quantization        (D)
skb/lm.py            common model interface + word beam search
skb/evaluate.py      perplexity, top-3 accuracy, size, latency          (D)
skb/models.py        loads the model ladder
scripts/upload_weights.py   push artifacts/ to the Hugging Face Hub
scripts/check_int8_cache.py  experiment: may the INT8 model use the KV cache?
tests/test_core.py   unit tests (normalization, n-gram probabilities, beam search)
docs/architecture.png       architecture diagram used in the presentation
```

## 8. AI usage

Claude (Anthropic) was used to help write the code, the README and to draft the slides. All models were
trained and all numbers were measured by me with `train_all.py` on my own computer.
