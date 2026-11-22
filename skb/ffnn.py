"""Area A: feed-forward neural language model (Bengio et al., 2003 style).

  previous 5 BPE tokens -> embedding (128) -> concat (640) -> hidden (512, tanh)
  -> dropout -> linear -> softmax over the 8,000-token BPE vocabulary

Trained from scratch with mini-batch SGD (momentum 0.9) and back-propagation,
cross-entropy loss, gradient clipping and a step learning-rate decay.
"""
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from . import config
from .lm import LanguageModel

FFNN_DIR = config.ART_DIR / "ffnn"


class FFNN(nn.Module):
    def __init__(self, vocab, context=config.FFNN_CONTEXT, emb=config.FFNN_EMB,
                 hidden=config.FFNN_HIDDEN, dropout=config.FFNN_DROPOUT):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab, emb)
        self.hidden = nn.Linear(context * emb, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x):                       # x: (B, context) token ids
        e = self.emb(x).flatten(1)
        h = self.drop(torch.tanh(self.hidden(e)))
        return self.out(h)                      # logits (B, V)


def make_examples(paragraph_ids, bos, C=config.FFNN_CONTEXT):
    xs, ys = [], []
    for ids in paragraph_ids:
        seq = np.asarray([bos] * C + list(ids), dtype=np.int64)
        if len(seq) <= C:
            continue
        win = np.lib.stride_tricks.sliding_window_view(seq, C + 1)
        xs.append(win[:, :C])
        ys.append(win[:, C])
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _token_ppl(model, X, Y, device, bs=4096):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs].to(device))
        tot += nn.functional.cross_entropy(logits, Y[i:i + bs].to(device), reduction="sum").item()
        n += len(Y[i:i + bs])
    return math.exp(tot / n)


def train(tok, train_ids, valid_ids, epochs=config.FFNN_EPOCHS, log=print):
    torch.manual_seed(config.SEED)
    device = pick_device()
    X, Y = make_examples(train_ids, tok.bos_id)
    Xv, Yv = make_examples(valid_ids, tok.bos_id)
    log(f"[ffnn] {len(X):,} training examples, device={device}")
    model = FFNN(tok.vocab_size).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=config.FFNN_LR, momentum=config.FFNN_MOMENTUM)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    history, best = [], math.inf
    FFNN_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        t0, run, steps = time.time(), [], 0
        perm = torch.randperm(len(X))
        for i in range(0, len(X), config.FFNN_BATCH):
            idx = perm[i:i + config.FFNN_BATCH]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()                      # back-propagation
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()                           # SGD update
            if steps % 50 == 0:
                run.append(loss.item())
            steps += 1
        sched.step()
        vppl = _token_ppl(model, Xv, Yv, device)
        tl = float(np.mean(run))
        history.append({"epoch": ep, "train_loss": tl, "valid_token_ppl": vppl, "seconds": time.time() - t0})
        log(f"[ffnn] epoch {ep}: train loss {tl:.3f}, valid token-PPL {vppl:.2f} ({time.time() - t0:.0f}s)")
        if vppl < best:
            best = vppl
            torch.save(model.state_dict(), FFNN_DIR / "ffnn.pt")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "ffnn_history.json").write_text(json.dumps(history, indent=2))
    return history


class FFNNLM(LanguageModel):
    name = "Feed-forward NN LM"
    precision = "FP32"

    def __init__(self, tok):
        self.tok = tok
        self.bos_id = tok.bos_id
        self.path = FFNN_DIR / "ffnn.pt"
        self.model = FFNN(tok.vocab_size)
        self.model.load_state_dict(torch.load(self.path, map_location="cpu"))
        self.model.eval()
        self.C = self.model.context

    def encode(self, text):
        return self.tok.encode(text)

    def piece_texts(self):
        return self.tok.piece_texts()

    def _ctx(self, seq):
        return ([self.bos_id] * self.C + list(seq))[-self.C:]

    @torch.no_grad()
    def next_logprobs(self, batch):
        x = torch.tensor([self._ctx(s) for s in batch], dtype=torch.long)
        return torch.log_softmax(self.model(x), -1).double().numpy()

    @torch.no_grad()
    def paragraph_logprob(self, text):
        ids = self.encode(text)
        if not ids:
            return 0.0
        X, Y = make_examples([ids], self.bos_id, self.C)
        lp = torch.log_softmax(self.model(X), -1)
        return float(lp.gather(1, Y[:, None]).sum())

    def size_mb(self):
        return self.path.stat().st_size / 2**20
