r"""
Text normalisation, vocabulary construction and hashed word n-gram features,
replicating the behaviour of the official fastText C++ code as closely as possible.

1. Normalisation (`normalize_text` in classification-results.sh)
   The official script lowercases each raw CSV line, prepends `__label__`, then runs:
       s/'/ ' /g ; s/"//g ; s/\./ \. /g ; s/<br \/>/ /g ; s/,/ , /g ; s/(/ ( /g ;
       s/)/ ) /g ; s/\!/ \! /g ; s/\?/ \? /g ; s/\;/ /g ; s/\:/ /g ; tr -s " "
   Because it works on the *raw CSV line* `"3","title","description"`, removing the
   quotes leaves `__label__3,title,description`, so the field separators become `,`
   tokens. We reproduce this exactly by normalising "," + title + "," + description.
   `tr '[:upper:]' '[:lower:]'` only lowercases ASCII (C locale) - we do the same.

2. Tokenisation (Dictionary::readWord)
   Split on whitespace; an end-of-sentence token "</s>" is appended to every line.

3. Vocabulary (Dictionary::readFromFile, -minCount 1)
   Built from the training split only; test/val words not in it are dropped as
   unigrams, but - exactly like fastText - their hashes still form bigrams.

4. Word n-grams (Dictionary::addWordNgrams) - the "hashing trick"
   Each token gets the 32-bit FNV-1a hash used by fastText (bytes are sign-extended
   from int8, as in the C++ code). For a bigram (w_i, w_{i+1}):
       h = int64(hash_i) * 116049371 + int64(hash_{i+1})     (mod 2**64)
       feature id = nwords + (h mod bucket)
   The int32 -> uint64 sign extension of the C++ code is reproduced too. A document is
   then the bag {unigram ids} U {n-gram bucket ids}, which the model averages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from configs.config import Config

EOS = "</s>"
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
NGRAM_MULT = np.uint64(116049371)

# The sed substitutions, in the same order as the official script.
_SED_RULES: Sequence[Tuple[str, str]] = (
    ("'", " ' "),
    ('"', ""),
    (".", " . "),
    ("<br />", " "),
    (",", " , "),
    ("(", " ( "),
    (")", " ) "),
    ("!", " ! "),
    ("?", " ? "),
    (";", " "),
    (":", " "),
)
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_WS = re.compile(r"[ \t\n\r\v\f\0]+")  # fastText's word delimiters


def normalize(text: str, cfg: Config) -> str:
    if cfg.lowercase:
        text = text.translate(_ASCII_LOWER)
    if cfg.fasttext_normalize:
        for old, new in _SED_RULES:
            text = text.replace(old, new)
    return text


def tokenize(title: str, description: str, cfg: Config) -> List[str]:
    """Normalise one AG News example exactly like the official pipeline and split it."""
    line = "," + title + "," + description  # what remains of the CSV line after s/"//g
    tokens = [t for t in _WS.split(normalize(line, cfg)) if t]
    if cfg.add_eos:
        tokens.append(EOS)
    return tokens


def fasttext_hash(token: str) -> int:
    """32-bit FNV-1a with int8 sign extension (Dictionary::hash). Returns signed int32."""
    h = 2166136261
    for b in token.encode("utf-8"):
        if b > 127:
            b = b - 256           # int8_t(c)
        h ^= b & 0xFFFFFFFF       # uint32_t(int8_t(c))
        h = (h * 16777619) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h  # stored as int32 in word_hashes


@dataclass
class Vocab:
    words: List[str]
    counts: List[int]

    def __post_init__(self):
        self.index: Dict[str, int] = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    @classmethod
    def build(cls, docs: Sequence[List[str]], min_count: int) -> "Vocab":
        counts: Dict[str, int] = {}
        for toks in docs:
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        # fastText sorts by descending frequency; ties broken alphabetically for determinism
        items = sorted(((w, c) for w, c in counts.items() if c >= min_count), key=lambda x: (-x[1], x[0]))
        return cls([w for w, _ in items], [c for _, c in items])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for w, c in zip(self.words, self.counts):
                f.write(f"{w}\t{c}\n")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        words, counts = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                w, c = line.rstrip("\n").rsplit("\t", 1)
                words.append(w); counts.append(int(c))
        return cls(words, counts)


@dataclass
class EncodedSplit:
    """A split as one flat feature array + per-document offsets (EmbeddingBag format)."""
    features: np.ndarray   # int64 [total_features]
    offsets: np.ndarray    # int64 [n_docs] start index of each document in `features`
    lengths: np.ndarray    # int64 [n_docs] number of features per document
    ntokens: np.ndarray    # int64 [n_docs] number of word tokens (for the LR schedule)
    labels: np.ndarray     # int64 [n_docs]

    def __len__(self) -> int:
        return len(self.labels)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, features=self.features, offsets=self.offsets, lengths=self.lengths,
                 ntokens=self.ntokens, labels=self.labels)

    @classmethod
    def load(cls, path: Path) -> "EncodedSplit":
        z = np.load(path)
        return cls(z["features"], z["offsets"], z["lengths"], z["ntokens"], z["labels"])


def encode(docs: Sequence[List[str]], labels: Sequence[int], vocab: Vocab,
           word_ngrams: int, bucket: int, hash_cache: Dict[str, int]) -> EncodedSplit:
    """Turn token lists into unigram ids + hashed n-gram ids (vectorised over the corpus)."""
    n_docs = len(docs)
    tok_counts = np.fromiter((len(d) for d in docs), dtype=np.int64, count=n_docs)
    doc_of_tok = np.repeat(np.arange(n_docs, dtype=np.int64), tok_counts)
    flat = [t for d in docs for t in d]

    # unigram ids (-1 for OOV, dropped below)
    idx = vocab.index
    uni = np.fromiter((idx.get(t, -1) for t in flat), dtype=np.int64, count=len(flat))

    pieces_ids = [uni]
    pieces_doc = [doc_of_tok]
    if word_ngrams > 1 and bucket > 0:
        hashes = np.empty(len(flat), dtype=np.int64)
        for i, t in enumerate(flat):
            h = hash_cache.get(t)
            if h is None:
                h = fasttext_hash(t)
                hash_cache[t] = h
            hashes[i] = h
        hashes_u = hashes.view(np.uint64)  # int32 -> int64 -> uint64 == C++ sign extension
        nwords = np.uint64(len(vocab))
        h = hashes_u.copy()
        with np.errstate(over="ignore"):
            for k in range(1, word_ngrams):
                # n-gram of length k+1 starting at position i: needs i+k in the same document
                h = h[:-1] * NGRAM_MULT + hashes_u[k:]
                start_doc = doc_of_tok[: len(h)]
                same = start_doc == doc_of_tok[k:]
                ids = (h[same] % np.uint64(bucket)) + nwords
                pieces_ids.append(ids.astype(np.int64))
                pieces_doc.append(start_doc[same])

    ids = np.concatenate(pieces_ids)
    dids = np.concatenate(pieces_doc)
    keep = ids >= 0
    ids, dids = ids[keep], dids[keep]
    order = np.argsort(dids, kind="stable")
    features = ids[order]
    lengths = np.bincount(dids, minlength=n_docs).astype(np.int64)
    offsets = np.zeros(n_docs, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]
    return EncodedSplit(features, offsets, lengths, tok_counts, np.asarray(labels, dtype=np.int64))


def _cache_key(cfg: Config) -> str:
    return (f"ng{cfg.word_ngrams}_b{cfg.effective_bucket}_mc{cfg.min_count}_val{cfg.val_fraction}"
            f"_seed{cfg.seed}_full{int(cfg.train_on_full_train)}_eos{int(cfg.add_eos)}"
            f"_norm{int(cfg.fasttext_normalize)}_lc{int(cfg.lowercase)}")


def prepare_data(cfg: Config, verbose: bool = True) -> Tuple[Vocab, Dict[str, EncodedSplit]]:
    """Load (from cache if possible) the encoded train/val/test splits and the vocabulary."""
    from src.data_loader import download_ag_news, load_ag_news

    cache = cfg.processed_dir / _cache_key(cfg)
    names = ("train", "val", "test")
    if all((cache / f"{n}.npz").exists() for n in names) and (cache / "vocab.txt").exists():
        if verbose:
            print(f"Loading cached features from {cache}")
        vocab = Vocab.load(cache / "vocab.txt")
        return vocab, {n: EncodedSplit.load(cache / f"{n}.npz") for n in names}

    download_ag_news(cfg)
    raw = load_ag_news(cfg)
    toks = {n: [tokenize(t, d, cfg) for _, t, d in raw[n]] for n in names}
    labels = {n: [lab for lab, _, _ in raw[n]] for n in names}
    vocab = Vocab.build(toks["train"], cfg.min_count)
    hash_cache: Dict[str, int] = {}
    enc = {n: encode(toks[n], labels[n], vocab, cfg.word_ngrams, cfg.effective_bucket, hash_cache)
           for n in names}
    cache.mkdir(parents=True, exist_ok=True)
    vocab.save(cache / "vocab.txt")
    for n in names:
        enc[n].save(cache / f"{n}.npz")
    if verbose:
        print(f"Vocabulary: {len(vocab)} words (incl. {EOS}); buckets: {cfg.effective_bucket}")
        for n in names:
            e = enc[n]
            print(f"{n:5s}: {len(e)} docs, {e.ntokens.mean():.1f} tokens/doc, "
                  f"{e.lengths.mean():.1f} features/doc")
        print("Example tokens:", toks["train"][0][:25])
        print(f"Cached to {cache}")
    return vocab, enc


if __name__ == "__main__":
    import argparse
    from configs.config import get_config

    ap = argparse.ArgumentParser(description="Preprocess AG News into fastText features")
    ap.add_argument("--word-ngrams", type=int, default=None)
    a = ap.parse_args()
    prepare_data(get_config(word_ngrams=a.word_ngrams))
