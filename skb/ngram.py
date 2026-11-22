"""Area B: n-gram language models over BPE tokens.

* AddOneNGram    - Laplace (add-one) smoothed trigram.
* KneserNeyNGram - interpolated Kneser-Ney (Chen & Goodman): absolute discount
  D = n1 / (n1 + 2 n2) per order, continuation counts for the lower orders and a
  uniform floor at the unigram level.

Counts are stored CSR-style: one dict  context -> (start, end, total, n_types)
plus two flat arrays (next-token ids sorted per context, counts). Pure NumPy.
"""
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config
from .lm import LanguageModel


class _Table:
    """All n-grams of one order, grouped by context."""

    def __init__(self, counter):
        groups = defaultdict(list)
        for gram, c in counter.items():
            groups[gram[:-1]].append((gram[-1], c))
        self.index = {}
        ws_all, cs_all = [], []
        pos = 0
        for ctx, items in groups.items():
            items.sort()
            ws_all.extend(w for w, _ in items)
            cs_all.extend(c for _, c in items)
            tot = float(sum(c for _, c in items))
            self.index[ctx] = (pos, pos + len(items), tot, len(items))
            pos += len(items)
        self.ws = np.asarray(ws_all, dtype=np.int32)
        self.cs = np.asarray(cs_all, dtype=np.float64)

    def get(self, ctx):
        e = self.index.get(ctx)
        if e is None:
            return None
        s, t, tot, types = e
        return self.ws[s:t], self.cs[s:t], tot, types

    def count(self, ctx, w):
        e = self.index.get(ctx)
        if e is None:
            return 0.0, None
        s, t, tot, types = e
        ws = self.ws[s:t]
        i = int(np.searchsorted(ws, w))
        c = float(self.cs[s + i]) if i < ws.size and ws[i] == w else 0.0
        return c, (tot, types)


def _discount(counter):
    n1 = sum(1 for c in counter.values() if c == 1)
    n2 = sum(1 for c in counter.values() if c == 2)
    return n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.5


def _count_ngrams(paragraph_ids, order, bos_id):
    c = Counter()
    for ids in paragraph_ids:
        seq = [bos_id] * (order - 1) + list(ids)
        c.update(zip(*[seq[i:] for i in range(order)]))
    return c


class _NGramBase(LanguageModel):
    precision = "float64 counts"

    def __init__(self, tokenizer, order=config.NGRAM_ORDER):
        self.tok = tokenizer
        self.order = order
        self.V = tokenizer.vocab_size
        self.bos_id = tokenizer.bos_id
        self.path = None

    def encode(self, text):
        return self.tok.encode(text)

    def piece_texts(self):
        return self.tok.piece_texts()

    def _hist(self, seq):
        n = self.order - 1
        return tuple(([self.bos_id] * n + list(seq))[-n:]) if n else ()

    def next_logprobs(self, batch):
        return np.log(np.stack([self.dist(self._hist(s)) for s in batch]))

    def paragraph_logprob(self, text):
        n = self.order - 1
        ids = [self.bos_id] * n + self.encode(text)
        return sum(math.log(self.prob(tuple(ids[i - n:i]), ids[i])) for i in range(n, len(ids)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k not in ("tok", "_mask_cache", "path")}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.path = path

    @classmethod
    def load(cls, path, tokenizer):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        obj.tok = tokenizer
        obj.path = Path(path)
        return obj

    def size_mb(self):
        return self.path.stat().st_size / 2**20


class AddOneNGram(_NGramBase):
    name = "Trigram (add-one)"

    def fit(self, paragraph_ids):
        self.table = _Table(_count_ngrams(paragraph_ids, self.order, self.bos_id))
        return self

    def dist(self, hist):
        e = self.table.get(hist)
        if e is None:
            return np.full(self.V, 1.0 / self.V)
        ws, cs, tot, _ = e
        p = np.full(self.V, 1.0 / (tot + self.V))
        p[ws] += cs / (tot + self.V)
        return p

    def prob(self, hist, w):
        c, info = self.table.count(hist, w)
        if info is None:
            return 1.0 / self.V
        return (c + 1.0) / (info[0] + self.V)


class KneserNeyNGram(_NGramBase):
    name = "Trigram (Kneser-Ney)"

    def fit(self, paragraph_ids):
        N = self.order
        counters = {N: _count_ngrams(paragraph_ids, N, self.bos_id)}
        for k in range(N - 1, 0, -1):               # continuation counts N1+(. ctx w)
            cont = Counter()
            for gram in counters[k + 1]:
                cont[gram[1:]] += 1
            counters[k] = cont
        self.D = {k: _discount(c) for k, c in counters.items()}
        self.tables = {k: _Table(c) for k, c in counters.items() if k > 1}
        # unigram level (continuation counts), interpolated with a uniform distribution
        uni = counters[1]
        tot, types, D1 = float(sum(uni.values())), len(uni), self.D[1]
        p1 = np.full(self.V, D1 * types / tot / self.V)
        for (w,), c in uni.items():
            p1[w] += max(c - D1, 0.0) / tot
        self.p1 = p1
        return self

    def dist(self, hist):
        p = self.p1.copy()
        for k in range(2, self.order + 1):
            e = self.tables[k].get(hist[len(hist) - (k - 1):])
            if e is None:
                continue
            ws, cs, tot, types = e
            D = self.D[k]
            p *= D * types / tot
            p[ws] += np.maximum(cs - D, 0.0) / tot
        return p

    def prob(self, hist, w):
        p = float(self.p1[w])
        for k in range(2, self.order + 1):
            c, info = self.tables[k].count(hist[len(hist) - (k - 1):], w)
            if info is None:
                continue
            tot, types = info
            D = self.D[k]
            p = D * types / tot * p + max(c - D, 0.0) / tot
        return p
