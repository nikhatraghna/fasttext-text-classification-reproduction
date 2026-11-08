"""
Training (with per-epoch validation, checkpointing and resume) and the
learning-rate selection sweep described in the paper.

Examples (run from the repository root):
    python -m src.train --sweep --word-ngrams 2        # pick LR from {0.05,0.1,0.25,0.5} on validation
    python -m src.train --word-ngrams 2 --lr best      # final bigram model with the selected LR
    python -m src.train --word-ngrams 1 --lr best      # final unigram model
    python -m src.train --word-ngrams 2 --lr 0.25      # official script's LR, no sweep

Optimisation (fastText, src/fasttext.cc):
  * plain SGD, no momentum / weight decay;
  * learning rate decays linearly from lr0 to 0 with the number of processed tokens:
        lr_t = lr0 * (1 - tokens_seen / (epochs * tokens_per_epoch));
  * per-example updates (Hogwild over threads). We use small mini-batches with a summed
    loss, so a step of size B approximates B consecutive per-example updates.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from configs.config import Config, default_run_name, get_config
from src.model import FastTextClassifier, count_parameters
from src.preprocessing import EncodedSplit, prepare_data
from src.utils import (environment_info, get_rng_state, load_json, save_csv, save_json,
                       set_rng_state, set_seed, setup_logger)


# --------------------------------------------------------------------------- batching
def make_batch(split: EncodedSplit, idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the documents `idx` into EmbeddingBag (features, offsets) + labels."""
    lengths = split.lengths[idx]
    starts = split.offsets[idx]
    batch_offsets = np.zeros(len(idx), dtype=np.int64)
    batch_offsets[1:] = np.cumsum(lengths)[:-1]
    total = int(lengths.sum())
    # position of every feature in the flat array, built without a Python loop
    pos = np.repeat(starts - batch_offsets, lengths) + np.arange(total, dtype=np.int64)
    return (torch.from_numpy(split.features[pos]),
            torch.from_numpy(batch_offsets),
            torch.from_numpy(split.labels[idx]))


@torch.no_grad()
def predict(model: FastTextClassifier, split: EncodedSplit, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (predicted labels, probabilities, mean cross-entropy) for a split."""
    model.eval()
    preds, probs, loss_sum = [], [], 0.0
    for s in range(0, len(split), batch_size):
        idx = np.arange(s, min(s + batch_size, len(split)))
        feats, offs, y = make_batch(split, idx)
        logits = model(feats, offs)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        p = torch.softmax(logits, dim=1)
        probs.append(p.numpy()); preds.append(p.argmax(1).numpy())
    model.train()
    return np.concatenate(preds), np.concatenate(probs), loss_sum / len(split)


def evaluate_split(model, split: EncodedSplit, batch_size: int) -> Dict[str, float]:
    pred, _, loss = predict(model, split, batch_size)
    return {"loss": loss, "acc": float((pred == split.labels).mean())}


# --------------------------------------------------------------------------- checkpoints
def _ckpt_dir(cfg: Config, run_name: str) -> Path:
    return Path(cfg.checkpoint_dir) / run_name


def save_checkpoint(path: Path, model, optimizer, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), **state}, tmp)
    tmp.replace(path)  # atomic rename: a disconnect mid-save never corrupts last.pt


def load_model(cfg: Config, run_name: str, vocab_size: int, which: str = "final") -> FastTextClassifier:
    ckpt = torch.load(_ckpt_dir(cfg, run_name) / f"{which}.pt", map_location="cpu", weights_only=False)
    model = FastTextClassifier.from_config(cfg, vocab_size)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- training loop
def train_model(cfg: Config, run_name: str, vocab_size: int, data: Dict[str, EncodedSplit],
                checkpoint: bool = True, logger=None) -> Tuple[FastTextClassifier, List[dict], dict]:
    log = logger.info if logger else print
    torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)

    train, val = data["train"], data["val"]
    model = FastTextClassifier.from_config(cfg, vocab_size)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=cfg.lr)  # supports the sparse embedding gradients

    tokens_per_epoch = int(train.ntokens.sum())
    total_tokens = cfg.epochs * tokens_per_epoch
    history: List[dict] = []
    start_epoch, tokens_seen, global_step = 1, 0, 0

    ckdir = _ckpt_dir(cfg, run_name)
    last, final = ckdir / "last.pt", ckdir / "final.pt"
    if checkpoint and cfg.resume and final.exists() and not last.exists():
        # A finished run: treat its final checkpoint as "completed epoch = epochs".
        ck = torch.load(final, map_location="cpu", weights_only=False)
        ck.setdefault("tokens_seen", total_tokens); ck.setdefault("global_step", 0)
        ck.setdefault("rng", get_rng_state())
        last_src = ck
    elif checkpoint and cfg.resume and last.exists():
        last_src = torch.load(last, map_location="cpu", weights_only=False)
    else:
        last_src = None
    if last_src is not None:
        ck = last_src
        same = all(ck["config"].get(k) == v for k, v in
                   (("word_ngrams", cfg.word_ngrams), ("bucket", cfg.bucket), ("lr", cfg.lr),
                    ("embedding_dim", cfg.embedding_dim), ("epochs", cfg.epochs),
                    ("batch_size", cfg.batch_size), ("seed", cfg.seed)))
        if same:
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            history, tokens_seen, global_step = ck["history"], ck["tokens_seen"], ck["global_step"]
            start_epoch = ck["epoch"] + 1
            set_rng_state(ck["rng"])
            log(f"Resumed '{run_name}' from {ckdir} (completed epoch {ck['epoch']}/{cfg.epochs})")
        else:
            log(f"Checkpoint in {ckdir} was made with a different config - starting from scratch")

    log(f"Run '{run_name}': {count_parameters(model):,} parameters "
        f"({vocab_size:,} words + {cfg.effective_bucket:,} buckets) x {cfg.embedding_dim} + output; "
        f"train={len(train)}, val={len(val)}, lr0={cfg.lr}, epochs={cfg.epochs}, batch={cfg.batch_size}")

    model.train()
    for epoch in range(start_epoch, cfg.epochs + 1):
        perm = np.random.permutation(len(train)) if cfg.shuffle_each_epoch else np.arange(len(train))
        t0 = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0
        for b in range(0, len(train), cfg.batch_size):
            idx = perm[b:b + cfg.batch_size]
            # linear LR decay by processed tokens, as in FastText::trainThread
            lr = cfg.lr * max(0.0, 1.0 - tokens_seen / total_tokens) if cfg.lr_schedule == "linear" else cfg.lr
            for g in optimizer.param_groups:
                g["lr"] = lr
            feats, offs, y = make_batch(train, idx)
            logits = model(feats, offs)
            loss = F.cross_entropy(logits, y, reduction=cfg.loss_reduction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = len(idx)
            tokens_seen += int(train.ntokens[idx].sum())
            global_step += 1
            loss_sum += loss.item() * (1 if cfg.loss_reduction == "sum" else n)
            correct += int((logits.argmax(1) == y).sum())
            seen += n
            if cfg.log_every_steps and global_step % cfg.log_every_steps == 0:
                log(f"  epoch {epoch} step {global_step} | lr {lr:.5f} | "
                    f"train loss {loss_sum / seen:.4f} | train acc {correct / seen:.4f}")
        epoch_time = time.perf_counter() - t0

        v = evaluate_split(model, val, cfg.eval_batch_size)
        rec = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "val_loss": v["loss"], "val_acc": v["acc"], "lr_end": lr,
               "epoch_time_sec": epoch_time, "global_step": global_step}
        history.append(rec)
        log(f"Epoch {epoch}/{cfg.epochs} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
            f"val loss {rec['val_loss']:.4f} acc {rec['val_acc']:.4f} | {epoch_time:.1f}s")

        if checkpoint and cfg.save_every_epoch:
            save_checkpoint(last, model, optimizer, {
                "epoch": epoch, "tokens_seen": tokens_seen, "global_step": global_step,
                "history": history, "rng": get_rng_state(), "config": cfg.to_dict(),
                "vocab_size": vocab_size})

    summary = {"run_name": run_name, "epochs": cfg.epochs, "lr": cfg.lr,
               "total_train_time_sec": float(sum(h["epoch_time_sec"] for h in history)),
               "mean_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history])) if history else None,
               "final_val_acc": history[-1]["val_acc"] if history else None,
               "num_parameters": count_parameters(model), "vocab_size": vocab_size,
               "buckets": cfg.effective_bucket, "train_docs": len(train), "val_docs": len(val),
               "tokens_per_epoch": tokens_per_epoch}
    if checkpoint and start_epoch <= cfg.epochs:  # skip re-saving an already finished run
        save_checkpoint(final, model, optimizer, {
            "epoch": cfg.epochs, "history": history, "config": cfg.to_dict(), "vocab_size": vocab_size,
            "tokens_seen": tokens_seen, "global_step": global_step, "rng": get_rng_state()})
        last.unlink(missing_ok=True)  # final.pt now holds everything needed to resume/skip
    return model, history, summary


# --------------------------------------------------------------------------- LR sweep
def ngram_tag(cfg: Config) -> str:
    return "bigram" if cfg.word_ngrams >= 2 else "unigram"


def run_sweep(cfg: Config) -> float:
    """Select lr0 from the paper's grid on the held-out validation split (resumable per LR)."""
    cfg.train_on_full_train = False
    tag = ngram_tag(cfg)
    logger = setup_logger(f"sweep_{tag}", cfg.logs_dir / f"lr_sweep_{tag}.log")
    vocab, data = prepare_data(cfg)
    sweep_dir = cfg.metrics_dir / "lr_sweep"
    rows = []
    for lr in cfg.lr_grid:
        out = sweep_dir / f"{tag}_lr{lr}.json"
        if out.exists():
            rec = load_json(out)
            logger.info(f"LR {lr}: already done (val acc {rec['val_acc']:.4f}) - skipping")
        else:
            run_cfg = dataclasses.replace(cfg, lr=lr)
            _, hist, summ = train_model(run_cfg, f"sweep_{tag}_lr{lr}", len(vocab), data,
                                        checkpoint=False, logger=logger)
            rec = {"word_ngrams": cfg.word_ngrams, "lr": lr, "val_acc": hist[-1]["val_acc"],
                   "val_loss": hist[-1]["val_loss"], "train_time_sec": summ["total_train_time_sec"]}
            save_json(rec, out)
        rows.append(rec)
    best = max(rows, key=lambda r: r["val_acc"])
    save_csv(rows, cfg.metrics_dir / f"lr_sweep_{tag}.csv")
    save_json({"word_ngrams": cfg.word_ngrams, "best_lr": best["lr"], "best_val_acc": best["val_acc"],
               "grid": list(cfg.lr_grid)}, cfg.metrics_dir / f"best_lr_{tag}.json")
    from src.utils import plot_lr_sweep
    plot_lr_sweep(rows, cfg.figures_dir / f"lr_sweep_{tag}.png",
                  f"LR selection on validation ({tag}, {cfg.epochs} epochs)")
    logger.info(f"Best LR for {tag}: {best['lr']} (val acc {best['val_acc']:.4f})")
    return best["lr"]


# --------------------------------------------------------------------------- CLI
def parse_args():
    ap = argparse.ArgumentParser(description="Train the fastText classifier on AG News")
    ap.add_argument("--word-ngrams", type=int, default=None, help="1 = unigrams, 2 = bigrams (default from config)")
    ap.add_argument("--lr", type=str, default=None, help="float, or 'best' to use the sweep result")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None, help="embedding / hidden size h")
    ap.add_argument("--bucket", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--full-train", action="store_true", help="train on all 120k rows (official script)")
    ap.add_argument("--sweep", action="store_true", help="run the LR grid search instead")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = get_config(word_ngrams=a.word_ngrams, epochs=a.epochs, embedding_dim=a.dim, bucket=a.bucket,
                     batch_size=a.batch_size, num_threads=a.threads, seed=a.seed)
    if a.dim is not None:
        cfg.init_uniform_scale = 1.0 / a.dim
    if a.full_train:
        cfg.train_on_full_train = True
    if a.no_resume:
        cfg.resume = False

    if a.sweep:
        run_sweep(cfg)
        return

    tag = ngram_tag(cfg)
    if a.lr == "best":
        best_file = cfg.metrics_dir / f"best_lr_{tag}.json"
        if best_file.exists():
            cfg.lr = float(load_json(best_file)["best_lr"])
            print(f"Using LR {cfg.lr} selected on validation ({best_file.name})")
        else:
            print(f"WARNING: {best_file} not found (run --sweep first); using config LR {cfg.lr}")
    elif a.lr is not None:
        cfg.lr = float(a.lr)

    run_name = a.run_name or default_run_name(cfg)
    if not cfg.resume:
        shutil.rmtree(_ckpt_dir(cfg, run_name), ignore_errors=True)
    logger = setup_logger(run_name, cfg.logs_dir / f"{run_name}.log")
    vocab, data = prepare_data(cfg)
    _, history, summary = train_model(cfg, run_name, len(vocab), data, checkpoint=True, logger=logger)

    summary["config"] = cfg.to_dict()
    summary["environment"] = environment_info()
    save_json(history, cfg.metrics_dir / f"{run_name}_history.json")
    save_csv(history, cfg.metrics_dir / f"{run_name}_history.csv")
    save_json(summary, cfg.metrics_dir / f"{run_name}_train_summary.json")
    from src.utils import plot_training_curves
    for p in plot_training_curves(history, cfg.figures_dir, run_name):
        logger.info(f"Saved {p}")
    logger.info(f"Done. Final model: {_ckpt_dir(cfg, run_name) / 'final.pt'}")


if __name__ == "__main__":
    main()
