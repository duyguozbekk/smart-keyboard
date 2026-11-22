"""Load the trained model ladder (used by evaluation and by the app)."""
from . import config
from .bpe import BPETokenizer
from .ffnn import FFNNLM
from .gpt2 import GPT2LM
from .ngram import AddOneNGram, KneserNeyNGram

NGRAM_DIR = config.ART_DIR / "ngram"
ADDONE_FILE = NGRAM_DIR / "trigram_addone.pkl"
KN_FILE = NGRAM_DIR / "trigram_kn.pkl"


def load_ladder(include_fp32_gpt2=True, log=print):
    """Return [(key, model)] from the simplest model to the deployed one (GPT-2 INT8 last)."""
    tok = BPETokenizer()
    models = []
    log("[load] n-gram models")
    models.append(("ngram_addone", AddOneNGram.load(ADDONE_FILE, tok)))
    models.append(("ngram_kn", KneserNeyNGram.load(KN_FILE, tok)))
    log("[load] feed-forward NN")
    models.append(("ffnn", FFNNLM(tok)))
    if include_fp32_gpt2:
        log("[load] GPT-2 FP32")
        models.append(("gpt2", GPT2LM(int8=False)))
    log("[load] GPT-2 INT8")
    models.append(("gpt2_int8", GPT2LM(int8=True)))
    for _, m in models:          # build word masks once so the first suggestion is fast
        m._masks()
    return models
