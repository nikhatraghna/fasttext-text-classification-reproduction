"""
AG News download, integrity checking, loading and train/validation splitting.

The files are the original Zhang et al. (2015) CSVs (`train.csv`, `test.csv`), each
row being: "class_index","title","description" with class_index in {1,2,3,4}.
These are the same files the official fastText script downloads.

Run directly to download:  python -m src.data_loader
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from configs.config import Config, get_config

Row = Tuple[int, str, str]  # (label 0..3, title, description)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fasttext-reproduction/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _try_csv_mirrors(cfg: Config) -> bool:
    for template in cfg.csv_mirrors:
        try:
            for split in ("train", "test"):
                url = template.format(split=split)
                print(f"Downloading {url}")
                data = _http_get(url)
                (cfg.raw_dir / f"{split}.csv").write_bytes(data)
            return True
        except Exception as e:  # noqa: BLE001 - try the next mirror
            print(f"  mirror failed: {e}")
    return False


def _try_gdrive_archive(cfg: Config) -> bool:
    """Fallback: original Google Drive tarball used by classification-results.sh."""
    url = ("https://drive.usercontent.google.com/download?id="
           f"{cfg.gdrive_archive_id}&export=download&confirm=t")
    try:
        print(f"Downloading {url}")
        data = _http_get(url, timeout=120)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for split in ("train", "test"):
                member = tar.getmember(f"ag_news_csv/{split}.csv")
                with tar.extractfile(member) as src, open(cfg.raw_dir / f"{split}.csv", "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  Google Drive fallback failed: {e}")
        return False


def verify_files(cfg: Config, strict: bool = False) -> Dict[str, bool]:
    """Check SHA-256 and row counts. Returns {split: ok}."""
    status = {}
    for split in ("train", "test"):
        p = cfg.raw_dir / f"{split}.csv"
        if not p.exists():
            status[split] = False
            continue
        digest_ok = sha256_of(p) == cfg.sha256[split]
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in csv.reader(f))
        rows_ok = n == cfg.expected_rows[split]
        status[split] = digest_ok and rows_ok
        msg = f"{split}.csv: rows={n} (expected {cfg.expected_rows[split]}), sha256 {'OK' if digest_ok else 'MISMATCH'}"
        print(msg)
        if strict and not status[split]:
            raise RuntimeError(f"Integrity check failed: {msg}")
    return status


def download_ag_news(cfg: Config, force: bool = False) -> Path:
    """Download AG News into data/raw/ag_news_csv/ (skips if already valid)."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    if not force and all(verify_files(cfg).values()):
        print(f"AG News already present and verified in {cfg.raw_dir}")
        return cfg.raw_dir
    if not (_try_csv_mirrors(cfg) or _try_gdrive_archive(cfg)):
        raise RuntimeError(
            "Could not download AG News. Download ag_news_csv.tar.gz manually (e.g. from the "
            "official fastText script's Google Drive link) and place train.csv / test.csv in "
            f"{cfg.raw_dir}")
    status = verify_files(cfg)
    if not all(status.values()):
        print("WARNING: downloaded files do not match the reference checksum/row count. "
              "Results may not be comparable.")
    return cfg.raw_dir


def load_split(cfg: Config, split: str) -> List[Row]:
    """Read one CSV split. Labels are converted from 1..4 to 0..3."""
    path = cfg.raw_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data_loader` first")
    rows: List[Row] = []
    with open(path, encoding="utf-8") as f:
        for rec in csv.reader(f):
            label = int(rec[0]) - 1
            title = rec[1] if len(rec) > 1 else ""
            desc = ",".join(rec[2:]) if len(rec) > 2 else ""
            rows.append((label, title, desc))
    return rows


def train_val_split(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratification is unnecessary (AG train is perfectly balanced: 30k/class);
    a seeded random permutation keeps the split deterministic."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def load_ag_news(cfg: Config) -> Dict[str, List[Row]]:
    """Return {'train', 'val', 'test'} lists of rows according to the config."""
    train_all = load_split(cfg, "train")
    test = load_split(cfg, "test")
    tr_idx, va_idx = train_val_split(len(train_all), cfg.val_fraction, cfg.seed)
    val = [train_all[i] for i in va_idx]
    train = train_all if cfg.train_on_full_train else [train_all[i] for i in tr_idx]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and verify AG News")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    cfg = get_config()
    download_ag_news(cfg, force=args.force)
    splits = load_ag_news(cfg)
    for k, v in splits.items():
        counts = np.bincount([r[0] for r in v], minlength=cfg.num_classes)
        print(f"{k:5s}: {len(v):6d} rows, per-class {counts.tolist()}")
    print("Example:", splits["train"][0])


if __name__ == "__main__":
    main()
