"""Corpus loading and regex normalization (area B: text normalization).

WikiText-2 (raw) is a ~2M-word subset of English Wikipedia "Good" and "Featured"
articles with fixed train / validation / test splits. The raw files are lightly
tokenized ("the game , which ..."), contain section headings (" = Title = ") and
the escape tokens @-@, @,@ and @.@. We undo that with regular expressions so the
keyboard sees normal running text.
"""
import json
import re

from . import config

HEADING_RE = re.compile(r"^\s*=+ .* =+\s*$")
ESCAPES = [(re.compile(r" @-@ "), "-"), (re.compile(r" @,@ "), ","), (re.compile(r" @\.@ "), ".")]
SPACE_BEFORE_PUNCT = re.compile(r" ([,.;:!?%)\]}])")
SPACE_AFTER_OPEN = re.compile(r"([(\[{$]) ")
CLITIC = re.compile(r" ('s|'re|'ve|'ll|'d|'m|n't)\b")
QUOTES = re.compile(r'" (.*?) "')
MULTISPACE = re.compile(r"\s+")


def normalize_line(line: str) -> str:
    """Return a cleaned paragraph, or '' for headings / empty lines."""
    if not line.strip() or HEADING_RE.match(line):
        return ""
    s = line.strip()
    for pat, rep in ESCAPES:
        s = pat.sub(rep, s)
    s = CLITIC.sub(r"\1", s)
    s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = SPACE_AFTER_OPEN.sub(r"\1", s)
    s = QUOTES.sub(r'"\1"', s)
    s = MULTISPACE.sub(" ", s).strip()
    return s


def normalize_lines(lines):
    out = []
    for line in lines:
        s = normalize_line(line)
        if s:
            out.append(s)
    return out


def prepare(force: bool = False):
    """Download WikiText-2, normalize it and write data/{train,valid,test}.txt."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {s: config.DATA_DIR / f"{s}.txt" for s in ("train", "valid", "test")}
    if not force and all(p.exists() for p in paths.values()):
        print("[data] already prepared")
        return stats()
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    split_map = {"train": "train", "valid": "validation", "test": "test"}
    for ours, theirs in split_map.items():
        paras = normalize_lines(ds[theirs]["text"])
        paths[ours].write_text("\n".join(paras) + "\n", encoding="utf-8")
    return stats()


def load_split(split: str):
    path = config.DATA_DIR / f"{split}.txt"
    return [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def stats():
    s = {}
    for split in ("train", "valid", "test"):
        paras = load_split(split)
        s[split] = {"paragraphs": len(paras), "words": sum(len(p.split()) for p in paras)}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "data_stats.json").write_text(json.dumps(s, indent=2))
    print("[data]", s)
    return s
