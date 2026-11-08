"""
OPTIONAL sanity baseline: train the *official* fastText library (pip package `fasttext`)
on exactly the same normalised training split and evaluate it on the same test set.

This separates "does our PyTorch re-implementation match fastText?" from "does fastText
on this machine match the 2016 paper?". It is not needed for the main pipeline and its
package is not in requirements.txt (install with `pip install fasttext`).

Uses the official AG News command: -dim 10 -lr <lr> -wordNgrams 2 -minCount 1
-bucket 10000000 -epoch 5 -thread <n>.

    python -m src.official_baseline            # lr from the bigram sweep if present, else 0.25
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from configs.config import get_config


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"

from src.data_loader import download_ag_news, load_ag_news
from src.evaluate import compute_metrics
from src.preprocessing import tokenize
from src.utils import environment_info, load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    try:
        import fasttext  # noqa: F401
    except ImportError:
        print("The `fasttext` package is not installed; run `pip install fasttext` to use this baseline.")
        return

    cfg = get_config(word_ngrams=2, num_threads=a.threads)
    best = cfg.metrics_dir / "best_lr_bigram.json"
    lr = a.lr if a.lr is not None else (float(load_json(best)["best_lr"]) if best.exists() else cfg.lr)
    download_ag_news(cfg)
    raw = load_ag_news(cfg)
    work = cfg.processed_dir / "official_fasttext"
    work.mkdir(parents=True, exist_ok=True)

    def write(split):
        path = work / f"{split}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for lab, t, d in raw[split]:
                toks = tokenize(t, d, cfg)
                if toks and toks[-1] == "</s>":
                    toks = toks[:-1]  # the library appends </s> itself
                f.write(f"__label__{lab + 1} " + " ".join(toks) + "\n")
        return path

    train_txt, test_txt = write("train"), write("test")
    t0 = time.perf_counter()
    try:
        model = fasttext.train_supervised(str(train_txt), dim=cfg.embedding_dim, lr=lr,
                                          wordNgrams=cfg.word_ngrams, minCount=cfg.min_count,
                                          bucket=cfg.bucket, epoch=cfg.epochs, thread=cfg.num_threads,
                                          seed=cfg.seed, verbose=0)
    except RuntimeError as e:
        print(f"Official fastText training failed ({e}); baseline skipped.")
        save_json({"status": "failed", "error": str(e), "lr": lr}, cfg.metrics_dir / "official_fasttext_bigram_status.json")
        return
    train_time = time.perf_counter() - t0

    lines = test_txt.read_text(encoding="utf-8").splitlines()
    y_true = np.array([int(l.split(" ", 1)[0].replace("__label__", "")) - 1 for l in lines])
    # call the C++ predictor directly (the Python wrapper's predict() breaks on NumPy 2)
    y_pred = np.array([int(model.f.predict(l.split(" ", 1)[1] + "\n", 1, 0.0, "strict")[0][1]
                           .replace("__label__", "")) - 1 for l in lines])
    m = compute_metrics(y_true, y_pred, cfg.class_names)

    # Feature-pipeline check: copy the official weights into our PyTorch model (remapping
    # word ids) and compare predictions. 100% agreement means our tokenisation, vocabulary
    # and n-gram hashing are identical to the C++ implementation.
    agreement = None
    try:
        import torch
        from src.model import FastTextClassifier
        from src.preprocessing import prepare_data
        from src.train import predict
        vocab, data = prepare_data(cfg, verbose=False)
        A, B = model.get_input_matrix(), model.get_output_matrix()
        their_nwords = len(model.get_words())
        if their_nwords == len(vocab):
            W = np.zeros((len(vocab) + cfg.effective_bucket, cfg.embedding_dim), dtype=np.float32)
            for i, w in enumerate(vocab.words):
                W[i] = A[model.get_word_id(w)]
            W[len(vocab):] = A[their_nwords:]
            Bo = np.zeros((cfg.num_classes, cfg.embedding_dim), dtype=np.float32)
            for j, lab in enumerate(model.get_labels()):
                Bo[int(lab.replace("__label__", "")) - 1] = B[j]
            ours = FastTextClassifier.from_config(cfg, len(vocab))
            ours.embedding.weight.data = torch.from_numpy(W)
            ours.fc.weight.data = torch.from_numpy(Bo)
            ours_pred, _, _ = predict(ours, data["test"])
            agreement = float((ours_pred == y_pred).mean())
            print(f"Prediction agreement (our model with official weights vs official): {agreement:.4f}")
        else:
            print(f"Vocabulary sizes differ (ours {len(vocab)}, official {their_nwords}); check skipped")
    except Exception as e:  # noqa: BLE001 - the check is informative only
        print(f"Feature-pipeline check skipped: {e}")

    m.update({"feature_pipeline_agreement": agreement, "run_name": "official_fasttext_bigram", "lr": lr, "total_train_time_sec": train_time,
              "mean_epoch_time_sec": train_time / cfg.epochs, "threads": cfg.num_threads,
              "fasttext_version": _pkg_version("fasttext"),
              "environment": environment_info()})
    save_json(m, cfg.metrics_dir / "official_fasttext_bigram_test_metrics.json")
    print(f"Official fastText (lr={lr}): test accuracy {m['accuracy']:.4f}, "
          f"train time {train_time:.1f}s for {cfg.epochs} epochs")


if __name__ == "__main__":
    main()
