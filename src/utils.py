"""
Shared utilities: seeding, logging, JSON/CSV I/O, environment info and plotting.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


# --------------------------------------------------------------------------- logging
def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Logger that writes to stdout and (optionally) appends to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- I/O
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)  # atomic: a disconnect never leaves a half-written file


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: Sequence[Dict], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def environment_info() -> dict:
    """Describe the compute environment (recorded with every run)."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:  # model name of the CPU on Linux (Colab)
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info["mem_total"] = f.readline().split(":", 1)[1].strip()
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------- plotting
def _plt():
    import matplotlib
    matplotlib.use("Agg")  # headless; figures are saved, then displayed in Colab
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(history: List[Dict], out_dir: Path, run_name: str) -> List[Path]:
    """Save loss curves and accuracy curves from a per-epoch history list."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    paths = []

    # Training loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", color="#1f77b4", label="train (running mean over epoch)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.set_title(f"Training loss — {run_name}")
    ax.set_xticks(epochs); ax.grid(alpha=0.3); ax.legend()
    p = out_dir / f"{run_name}_train_loss.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Validation loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", color="#d62728", label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.set_title(f"Validation loss — {run_name}")
    ax.set_xticks(epochs); ax.grid(alpha=0.3); ax.legend()
    p = out_dir / f"{run_name}_val_loss.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Combined loss (handy for the report)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", color="#1f77b4", label="train (running mean over epoch)")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="s", color="#d62728", label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.set_title(f"Loss — {run_name}")
    ax.set_xticks(epochs); ax.grid(alpha=0.3); ax.legend()
    p = out_dir / f"{run_name}_loss.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [100 * h["train_acc"] for h in history], marker="o", color="#1f77b4", label="train (running mean over epoch)")
    ax.plot(epochs, [100 * h["val_acc"] for h in history], marker="s", color="#d62728", label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"Accuracy — {run_name}")
    ax.set_xticks(epochs); ax.grid(alpha=0.3); ax.legend()
    p = out_dir / f"{run_name}_accuracy.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def plot_confusion_matrix(cm: np.ndarray, class_names: Iterable[str], out_path: Path,
                          title: str = "Confusion matrix", normalize: bool = False) -> Path:
    plt = _plt()
    class_names = list(class_names)
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(shown, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title)
    thresh = shown.max() / 2.0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if shown[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_lr_sweep(rows: List[Dict], out_path: Path, title: str) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    lrs = [r["lr"] for r in rows]
    ax.plot(lrs, [100 * r["val_acc"] for r in rows], marker="o", color="#2ca02c")
    ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([str(l) for l in lrs])
    ax.set_xlabel("Initial learning rate"); ax.set_ylabel("Validation accuracy (%)"); ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
