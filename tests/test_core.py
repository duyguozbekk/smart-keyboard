"""Pure-Python tests (no torch needed): normalization, n-gram probabilities, beam search."""
import math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from skb.data import normalize_line
from skb.ngram import AddOneNGram, KneserNeyNGram
from skb.lm import next_words


class WordPieceTok:
    """Tiny stand-in tokenizer: ' word' pieces + punctuation, id 0 = <s>."""
    def __init__(self, texts):
        import re
        self.re = re.compile(r" ?[A-Za-z]+|[^A-Za-z ]")
        vocab = {"": 0}
        for t in texts:
            for p in self.re.findall(t):
                vocab.setdefault(p, len(vocab))
        self.vocab = vocab; self.inv = list(vocab); self.vocab_size = len(vocab); self.bos_id = 0
    def encode(self, t): return [self.vocab.get(p, 1) for p in self.re.findall(t)]
    def piece_texts(self): return self.inv


def test_normalize():
    raw = ' The game \'s sales reached 1 @,@ 000 @-@ 2 @.@ 5 units ( worldwide ) . '
    assert normalize_line(raw) == "The game's sales reached 1,000-2.5 units (worldwide)."
    assert normalize_line(" = = Plot = = ") == ""


def test_ngram_and_beam():
    corpus = ["the cat sat on the mat.", "the cat ate the fish.", "a dog sat on the log.",
              "the dog ate the bone.", "the cat sat on the sofa."]
    tok = WordPieceTok(corpus)
    ids = [tok.encode(t) for t in corpus]
    for cls in (AddOneNGram, KneserNeyNGram):
        m = cls(tok).fit(ids)
        for hist in [(0, 0), (tok.vocab[" the"], tok.vocab[" cat"]), (5, 3)]:
            d = m.dist(hist)
            assert abs(d.sum() - 1) < 1e-9, (cls, d.sum())
            for w in range(tok.vocab_size):
                assert abs(d[w] - m.prob(hist, w)) < 1e-12
        lp = m.paragraph_logprob("the cat sat on the mat.")
        assert lp < 0 and math.isfinite(lp)
        with tempfile.TemporaryDirectory() as d:
            m.save(os.path.join(d, "m.pkl")); m2 = cls.load(os.path.join(d, "m.pkl"), tok)
            assert abs(m2.paragraph_logprob("the cat ate the fish.") - m.paragraph_logprob("the cat ate the fish.")) < 1e-9
            assert m2.size_mb() > 0
        sugg = next_words(m2, "the cat sat on the", k=3)
        print(cls.name, sugg)
        assert len(sugg) == 3 and all(w for w, _ in sugg)
    kn = KneserNeyNGram(tok).fit(ids)
    assert next_words(kn, "the cat sat on the")[0][0] in {"mat", "sofa", "log"}
    assert next_words(kn, "the cat")[0][0] == "sat"
    assert next_words(kn, "")  # empty context works




def test_multipiece_words():
    """Words split into several sub-word pieces are re-assembled by the beam search."""
    class SplitTok(WordPieceTok):
        def encode(self, t):
            out = []
            for p in self.re.findall(t):
                if p == " elephant":
                    out += [self.vocab[" ele"], self.vocab["phant"]]
                else:
                    out.append(self.vocab[p])
            return out
    corpus = ["the big elephant ran.", "the big elephant ate.", "the big dog ran."]
    tok = SplitTok(corpus)
    for p in (" ele", "phant"):
        tok.vocab[p] = len(tok.vocab); tok.inv.append(p)
    tok.vocab_size = len(tok.vocab)
    kn = KneserNeyNGram(tok).fit([tok.encode(t) for t in corpus[:3]])
    words = [w for w, _ in next_words(kn, "the big")]
    assert words[0] == "elephant", words


if __name__ == "__main__":
    test_normalize(); test_ngram_and_beam(); test_multipiece_words(); print("ALL TESTS PASSED")
