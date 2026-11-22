"""Area C (GPT-2 decoder fine-tuning) and area D (dynamic INT8 quantization).

Fine-tuning: GPT-2 small (124M) is trained further on the WikiText-2 training
paragraphs with the causal-LM objective (AdamW, linear warm-up/decay, 1 epoch,
blocks of 256 tokens). GPT-2 uses its own byte-level BPE (50,257 tokens).

Quantization: Hugging Face's GPT-2 implements its projections with `Conv1D`
(a transposed Linear). PyTorch dynamic quantization only recognises nn.Linear,
so every Conv1D is first converted into an equivalent nn.Linear. Then
`torch.ao.quantization.quantize_dynamic` stores the Linear weights of all 12
transformer blocks (attention + MLP) as INT8 and quantizes activations on the
fly. Embeddings, LayerNorms and the (embedding-tied) LM head stay FP32.
The quantized model runs on CPU.

Inference uses a KV cache: the context is encoded once, and the beam search
only feeds the few new sub-word tokens of each candidate word.
"""
import copy
import json
import math
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from . import config
from .lm import LanguageModel

FT_DIR = config.ART_DIR / "gpt2-finetuned"
Q_DIR = config.ART_DIR / "gpt2-int8"
Q_FILE = Q_DIR / "gpt2_int8_state_dict.pt"
INT8_CACHE_FILE = config.RESULTS_DIR / "int8_cache_check.json"   # written by scripts/check_int8_cache.py


def _tokenizer():
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(config.GPT2_BASE)
    tok.model_max_length = 10**9                 # we window long paragraphs ourselves
    return tok


def _stream(tok, paragraphs):
    eos = tok.eos_token_id
    ids = []
    for enc in tok(paragraphs)["input_ids"]:
        ids.append(eos)                          # same "<|endoftext|> + paragraph" format as in scoring
        ids.extend(enc)
    return ids


def finetune(train_paras, valid_paras, max_steps=None, log=print):
    from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
    from .ffnn import pick_device

    torch.manual_seed(config.SEED)
    device = pick_device()
    tok = _tokenizer()
    model = GPT2LMHeadModel.from_pretrained(config.GPT2_BASE).to(device)

    B, L = config.GPT2_BATCH, config.GPT2_BLOCK
    ids = torch.tensor(_stream(tok, train_paras))
    n_blocks = len(ids) // L
    blocks = ids[: n_blocks * L].view(n_blocks, L)
    vids = torch.tensor(_stream(tok, valid_paras))
    vblocks = vids[: (len(vids) // L) * L].view(-1, L)
    steps_per_epoch = math.ceil(n_blocks / B)
    total = steps_per_epoch * config.GPT2_EPOCHS
    if max_steps:
        total = min(total, max_steps)
    log(f"[gpt2] {len(ids):,} train tokens, {n_blocks} blocks, {total} steps, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=config.GPT2_LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, min(config.GPT2_WARMUP, total // 10 + 1), total)

    @torch.no_grad()
    def valid_loss(max_blocks=200):
        model.eval()
        vb = vblocks[:max_blocks]
        losses = [model(vb[i:i + B].to(device), labels=vb[i:i + B].to(device)).loss.item()
                  for i in range(0, len(vb), B)]
        model.train()
        return float(np.mean(losses))

    history = [{"step": 0, "valid_loss": valid_loss()}]
    log(f"[gpt2] step 0 valid loss {history[0]['valid_loss']:.3f}")
    model.train()
    step, t0 = 0, time.time()
    for ep in range(config.GPT2_EPOCHS):
        perm = torch.randperm(n_blocks)
        for i in range(0, n_blocks, B):
            xb = blocks[perm[i:i + B]].to(device)
            loss = model(xb, labels=xb).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                log(f"[gpt2] step {step}/{total} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
            if step >= total:
                break
        if step >= total:
            break
    history.append({"step": step, "valid_loss": valid_loss(), "seconds": time.time() - t0})
    log(f"[gpt2] done: valid loss {history[0]['valid_loss']:.3f} -> {history[-1]['valid_loss']:.3f}")
    FT_DIR.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(FT_DIR)
    tok.save_pretrained(FT_DIR)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "gpt2_history.json").write_text(json.dumps(history, indent=2))
    return history


# ----------------------------------------------------------------------------- quantization
def conv1d_to_linear(model):
    from transformers.pytorch_utils import Conv1D

    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                nx, nf = child.weight.shape     # Conv1D weight is (in, out)
                lin = nn.Linear(nx, nf)
                lin.weight.data = child.weight.data.t().contiguous()
                lin.bias.data = child.bias.data.clone()
                setattr(parent, name, lin)
    return model


def _set_engine():
    engines = torch.backends.quantized.supported_engines
    for e in (("qnnpack",) if platform.machine().lower() in ("arm64", "aarch64") else ()) + ("x86", "fbgemm", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


def _quantize(model):
    from torch.ao.quantization import quantize_dynamic

    _set_engine()
    model = conv1d_to_linear(model).eval()
    # Only the transformer body (attention + MLP Linear layers) is quantized. The LM head
    # shares its weight with the token embedding and stays FP32: it is the most
    # outlier-sensitive layer (one per-tensor scale for 50,257 output rows).
    # inplace=True keeps the weight tying, so the embedding is stored only once.
    quantize_dynamic(model.transformer, {nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


def quantize(log=print):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(FT_DIR).eval()
    qmodel = _quantize(model)
    Q_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), Q_FILE)
    model.config.save_pretrained(Q_DIR)
    log(f"[quant] engine={torch.backends.quantized.engine}, saved {Q_FILE.stat().st_size / 2**20:.1f} MB")


def load_int8():
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config.from_pretrained(Q_DIR)
    qmodel = _quantize(GPT2LMHeadModel(cfg).eval())
    qmodel.load_state_dict(torch.load(Q_FILE, map_location="cpu", weights_only=False))
    return qmodel.eval()


# ----------------------------------------------------------------------------- inference wrapper
def _lcp_len(seqs):
    n = min(len(x) for x in seqs)
    for i in range(n):
        t = seqs[0][i]
        if any(x[i] != t for x in seqs[1:]):
            return i
    return n


def _expand_past(past, B):
    """Copy the cached keys/values of the shared context for B beams."""
    if isinstance(past, (tuple, list)):                       # legacy tuple format
        return tuple(tuple(t.expand(B, *t.shape[1:]).contiguous() for t in layer) for layer in past)
    c = copy.deepcopy(past)                                    # transformers Cache object
    if B > 1:
        c.batch_repeat_interleave(B)
    return c


class GPT2LM(LanguageModel):
    WINDOW = 512

    def __init__(self, int8=False):
        from transformers import GPT2LMHeadModel

        self.int8 = int8
        self.tok = _tokenizer()
        self.bos_id = self.tok.eos_token_id
        if int8:
            self.model = load_int8()
            self.name, self.precision = "GPT-2 fine-tuned INT8", "INT8 dynamic (blocks), FP32 head"
        else:
            self.model = GPT2LMHeadModel.from_pretrained(FT_DIR).eval()
            self.name, self.precision = "GPT-2 fine-tuned", "FP32"
        self._pieces = None
        self._cache = None
        self.use_cache = True
        self.cache_check = self._check_cache()

    def encode(self, text):
        return self.tok(text)["input_ids"]

    def piece_texts(self):
        if self._pieces is None:
            V = self.model.config.vocab_size
            self._pieces = [self.tok.decode([i]) for i in range(V)]
        return self._pieces

    # -- next-token distribution -------------------------------------------------
    def _head(self, hidden_last):
        return torch.log_softmax(self.model.lm_head(hidden_last).float(), -1)

    @torch.no_grad()
    def _full_next(self, batch):
        """Reference path: run the whole sequences (no cache)."""
        h = self.model.transformer(torch.tensor(batch, dtype=torch.long)).last_hidden_state[:, -1]
        return self._head(h).double().numpy()

    @torch.no_grad()
    def _cached_next(self, batch):
        """KV-cache path: encode the shared context once, then only the new tokens."""
        B, first = len(batch), batch[0]
        p = _lcp_len(batch)
        c = self._cache
        if c is None or len(c[0]) > p or tuple(first[:len(c[0])]) != c[0]:
            prefix = first[:p]
            out = self.model.transformer(torch.tensor([prefix], dtype=torch.long), use_cache=True)
            c = self._cache = (tuple(prefix), out.past_key_values, self._head(out.last_hidden_state[:, -1]))
        ids, past, last = c
        if len(first) == len(ids):
            return last.expand(B, -1).double().numpy()
        rest = torch.tensor([s[len(ids):] for s in batch], dtype=torch.long)
        out = self.model.transformer(rest, past_key_values=_expand_past(past, B), use_cache=True)
        return self._head(out.last_hidden_state[:, -1]).double().numpy()

    def next_logprobs(self, batch):
        if self.use_cache:
            try:
                return self._cached_next(batch)
            except Exception as e:  # noqa: BLE001  (older/newer transformers cache API)
                print(f"[gpt2] KV cache disabled: {e}")
                self.use_cache = False
        return self._full_next(batch)

    def _check_cache(self):
        """Verify that the cached path gives the same distribution as the full path."""
        ctx = [self.bos_id] + self.encode("The album debuted at number one on the")
        batch = [ctx + self.encode(w) for w in (" Billboard", " UK", " US")]
        try:
            self._cache = None
            self._cached_next([ctx])
            got, ref = self._cached_next(batch), self._full_next(batch)
            top = np.argsort(-ref, axis=1)[:, :50]                # compare the tokens that matter
            diff = float(np.abs(np.take_along_axis(got, top, 1) - np.take_along_axis(ref, top, 1)).max())
        except Exception as e:  # noqa: BLE001
            print(f"[gpt2] KV cache unavailable ({e}); using full recomputation")
            self.use_cache = False
            return None
        if self.int8:
            # Dynamic INT8 recomputes activation scales on every call, so the cached path
            # (fewer tokens per call) gives slightly different log-probs by design. What must
            # match is the ranking: same top-1 token and >= 8 of the top-10 tokens in every row.
            top1 = bool((got.argmax(1) == ref.argmax(1)).all())
            g10, r10 = np.argsort(-got, 1)[:, :10], np.argsort(-ref, 1)[:, :10]
            overlap = min(len(set(a) & set(b)) for a, b in zip(g10, r10)) / 10
            ok = top1 and overlap >= 0.8
            self.cache_check_detail = {"maxdiff_top50": diff, "top1_equal": top1, "min_top10_overlap": overlap}
            if INT8_CACHE_FILE.exists():
                # data-driven decision: is the cached INT8 path at least as close to FP32
                # as the no-cache path on 200 test positions? (scripts/check_int8_cache.py)
                decision = json.loads(INT8_CACHE_FILE.read_text())
                ok = bool(decision.get("cache_allowed"))
                self.cache_check_detail["decided_by"] = "results/int8_cache_check.json"
        else:
            ok = diff <= 2e-3
            self.cache_check_detail = {"maxdiff_top50": diff}
        if not ok:
            print(f"[gpt2] KV cache check failed {self.cache_check_detail}; using full recomputation")
            self.use_cache = False
        self._cache = None
        return diff

    @torch.no_grad()
    def paragraph_logprob(self, text):
        ids = [self.bos_id] + self.encode(text)
        total, W = 0.0, self.WINDOW
        for s in range(0, len(ids) - 1, W - 1):
            win = torch.tensor([ids[s:s + W]])
            lp = torch.log_softmax(self.model(win).logits[0, :-1].float(), -1)
            total += float(lp.gather(1, win[0, 1:, None]).sum())
        return total

    def size_mb(self):
        if self.int8:
            return Q_FILE.stat().st_size / 2**20
        return sum(p.stat().st_size for p in FT_DIR.glob("*.safetensors")) / 2**20 or \
            sum(p.stat().st_size for p in FT_DIR.glob("*.bin")) / 2**20
