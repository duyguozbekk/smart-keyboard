"""Area B: byte-level BPE tokenizer trained on the training corpus.

Byte-level BPE never produces an unknown token, so every model can score the
exact same test text and word-level perplexities are comparable.
"""
from . import config

BPE_DIR = config.ART_DIR / "bpe"


def train(paragraphs=None):
    from tokenizers import ByteLevelBPETokenizer

    BPE_DIR.mkdir(parents=True, exist_ok=True)
    tok = ByteLevelBPETokenizer(add_prefix_space=False)
    train_file = str(config.DATA_DIR / "train.txt")
    tok.train(files=[train_file], vocab_size=config.BPE_VOCAB_SIZE, min_frequency=2,
              special_tokens=[config.BOS])
    tok.save_model(str(BPE_DIR))
    print(f"[bpe] trained, vocab={tok.get_vocab_size()}")
    return BPETokenizer()


class BPETokenizer:
    def __init__(self):
        from tokenizers import ByteLevelBPETokenizer

        self.tok = ByteLevelBPETokenizer(str(BPE_DIR / "vocab.json"), str(BPE_DIR / "merges.txt"),
                                         add_prefix_space=False)
        self.vocab_size = self.tok.get_vocab_size()
        self.bos_id = self.tok.token_to_id(config.BOS)
        self._pieces = None

    def encode(self, text):
        return self.tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts)]

    def piece_texts(self):
        if self._pieces is None:
            self._pieces = [self.tok.decode([i]) if i != self.bos_id else "" for i in range(self.vocab_size)]
        return self._pieces
