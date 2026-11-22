"""Common interface for all language models + word-level next-word suggestion.

Every model in the ladder (n-gram, feed-forward NN, GPT-2, GPT-2 INT8) works on
sub-word tokens. A keyboard must suggest *whole words*, so `next_words` runs a
small beam search over sub-word continuations:

  step 0  : expand tokens that start a new word  (" the", " Brit", ...)
  step 1+ : expand tokens that continue the word ("ish", "ain", ...)
  a word is finished with the probability mass of all "boundary" tokens
  (tokens that start with a space or punctuation).

Probabilities of identical words reached through different paths are summed.
"""
import math
import re
import time

import numpy as np

from . import config

WORD_START = re.compile(r"^ [A-Za-z]+$")
WORD_CONT = re.compile(r"^[A-Za-z]+$")


class LanguageModel:
    name = "base"
    precision = "-"
    bos_id = 0

    # --- to implement -------------------------------------------------------
    def encode(self, text: str):
        raise NotImplementedError

    def piece_texts(self):
        """Decoded text of every token id (index = id)."""
        raise NotImplementedError

    def next_logprobs(self, batch):
        """batch: list of equal-length token-id lists (BOS included).
        Returns np.ndarray (B, V) of natural-log next-token probabilities."""
        raise NotImplementedError

    def paragraph_logprob(self, text: str) -> float:
        """Sum of natural-log probabilities of all tokens of `text`, given BOS."""
        raise NotImplementedError

    def size_mb(self) -> float:
        raise NotImplementedError

    # --- shared helpers -----------------------------------------------------
    def _masks(self):
        if not hasattr(self, "_mask_cache"):
            texts = self.piece_texts()
            start = np.array([bool(WORD_START.match(t)) for t in texts])
            cont = np.array([bool(WORD_CONT.match(t)) for t in texts])
            boundary = np.array([len(t) > 0 and not t[0].isalnum() for t in texts])
            self._mask_cache = (start, cont, boundary, texts)
        return self._mask_cache

    def context_ids(self, context: str):
        ctx = context.rstrip()
        ids = self.encode(ctx) if ctx else []
        return [self.bos_id] + ids[-config.MAX_CONTEXT_TOKENS:]


def _logsumexp(x):
    if x.size == 0:
        return -math.inf
    m = float(np.max(x))
    if m == -math.inf:
        return m
    return m + math.log(float(np.sum(np.exp(x - m))))


def next_words(lm: LanguageModel, context: str, k: int = config.TOPK_WORDS,
               beam: int = config.BEAM, max_pieces: int = config.MAX_WORD_PIECES):
    """Return the top-k next words as [(word, probability), ...]."""
    start, cont, boundary, texts = lm._masks()
    base = lm.context_ids(context)
    beams = [([], 0.0, "")]          # (extra ids, log-prob, word text)
    finished = {}

    def add(word, lp):
        finished[word] = np.logaddexp(finished[word], lp) if word in finished else lp

    for step in range(max_pieces):
        logps = lm.next_logprobs([base + ext for ext, _, _ in beams])
        allowed = start if step == 0 else cont
        cands = []
        for (ext, lp, word), row in zip(beams, logps):
            if step > 0:                                  # word may end here
                add(word, lp + _logsumexp(row[boundary]))
            masked = np.where(allowed, row, -np.inf)
            b = min(beam, masked.size - 1)
            top = np.argpartition(-masked, b)[:b]
            for t in top:
                if masked[t] > -np.inf:
                    cands.append((ext + [int(t)], lp + float(masked[t]), word + texts[t]))
        cands.sort(key=lambda c: -c[1])
        beams = cands[:beam]
        if not beams:
            break
    for ext, lp, word in beams:                          # length limit reached
        add(word, lp)

    best = sorted(finished.items(), key=lambda kv: -kv[1])[:k]
    return [(w.strip(), float(math.exp(lp))) for w, lp in best]


def timed_next_words(lm, context, **kw):
    t0 = time.perf_counter()
    out = next_words(lm, context, **kw)
    return out, (time.perf_counter() - t0) * 1000.0
