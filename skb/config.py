"""Central configuration: paths and hyper-parameters used by every component."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # normalized corpus (git-ignored)
ART_DIR = ROOT / "artifacts"        # trained weights (git-ignored, hosted on HF Hub)
RESULTS_DIR = ROOT / "results"      # metrics + figures (committed)
if os.environ.get("SKB_QUICK"):     # smoke-test mode never overwrites the real artifacts/results
    ART_DIR = ROOT / "artifacts_quick"
    RESULTS_DIR = ROOT / "results_quick"

# Hugging Face Hub repo that hosts the trained weights (see scripts/upload_weights.py)
WEIGHTS_REPO = "duyguozbekk/smart-keyboard-weights"

SEED = 42

# --- Corpus -----------------------------------------------------------------
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

# --- B: BPE tokenizer -------------------------------------------------------
BPE_VOCAB_SIZE = 8000
BOS = "<s>"                         # paragraph start symbol for our own models

# --- B: n-gram language models ----------------------------------------------
NGRAM_ORDER = 3                     # trigram over BPE tokens

# --- A: feed-forward neural LM (Bengio et al. 2003 style) --------------------
FFNN_CONTEXT = 5                    # previous BPE tokens fed to the network
FFNN_EMB = 128
FFNN_HIDDEN = 512
FFNN_DROPOUT = 0.2
FFNN_EPOCHS = 4
FFNN_BATCH = 256
FFNN_LR = 0.05                      # plain SGD + momentum
FFNN_MOMENTUM = 0.9

# --- C: GPT-2 decoder --------------------------------------------------------
GPT2_BASE = "gpt2"                  # GPT-2 small, 124M parameters
GPT2_BLOCK = 256
GPT2_BATCH = 8
GPT2_LR = 5e-5
GPT2_EPOCHS = 1
GPT2_WARMUP = 100

# --- Inference / evaluation ---------------------------------------------------
TOPK_WORDS = 3
BEAM = 6                            # beam width when assembling whole words from sub-word tokens
MAX_WORD_PIECES = 4                 # max sub-word tokens per suggested word
MAX_CONTEXT_TOKENS = 128            # context window used for suggestions
EVAL_TOP3_SAMPLES = 1000            # test positions for top-3 next-word accuracy
LATENCY_RUNS = 200                  # >=100 runs required by the homework
