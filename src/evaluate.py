"""
Test-set evaluation, metric export, figures and the paper-vs-reproduction comparison.

Examples (from the repository root):
    python -m src.evaluate --word-ngrams 2            # evaluate checkpoints/fasttext_h10_bigram/final.pt
    python -m src.evaluate --word-ngrams 1
    python -m src.evaluate --compare                  # build results/metrics/comparison.{csv,md,json}

Outputs (results/):
    metrics/<run>_test_metrics.json        accuracy, macro/weighted P/R/F1, per-class scores, CM
    metrics/<run>_per_class.csv            per-class precision / recall / F1 / support
    metrics/<run>_classification_report.txt
    metrics/<run>_confusion_matrix.csv
    figures/<run>_confusion_matrix.png (+ _normalized.png)
    figures/<run>_{train_loss,val_loss,loss,accuracy}.png   (re-generated from history)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from configs.config import Config, default_run_name, get_config
from src.preprocessing import prepare_data
from src.train import load_model, ngram_tag, predict
from src.utils import (load_json, plot_confusion_matrix, plot_training_curves, save_csv,
                       save_json, setup_logger)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> dict:
    labels = list(range(len(class_names)))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "error_rate": float(1 - accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "per_class": [{"class": c, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for c, p, r, f, s in zip(class_names, p_c, r_c, f_c, s_c)],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                       target_names=list(class_names), digits=4),
    }


# Fields that determine the features / model shape; restored from the checkpoint so a
# model trained with e.g. --full-train or --dim 20 is evaluated with matching features.
_SHAPE_FIELDS = ("word_ngrams", "bucket", "embedding_dim", "min_count", "val_fraction", "seed",
                 "train_on_full_train", "add_eos", "lowercase", "fasttext_normalize")


def config_from_checkpoint(cfg: Config, run_name: str) -> Config:
    path = Path(cfg.checkpoint_dir) / run_name / "final.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train the model first (python -m src.train ...)")
    saved = torch.load(path, map_location="cpu", weights_only=False)["config"]
    for k in _SHAPE_FIELDS:
        if k in saved:
            setattr(cfg, k, saved[k])
    return cfg


def evaluate_run(cfg: Config, run_name: str) -> dict:
    cfg = config_from_checkpoint(cfg, run_name)
    logger = setup_logger(f"eval_{run_name}", cfg.logs_dir / f"{run_name}_eval.log")
    vocab, data = prepare_data(cfg, verbose=False)
    model = load_model(cfg, run_name, len(vocab), "final")
    t0 = time.perf_counter()
    y_pred, _, test_loss = predict(model, data["test"], cfg.eval_batch_size)
    infer_time = time.perf_counter() - t0
    y_true = data["test"].labels
    m = compute_metrics(y_true, y_pred, cfg.class_names)
    m.update({"run_name": run_name, "test_loss": test_loss, "test_docs": int(len(y_true)),
              "inference_time_sec": infer_time, "word_ngrams": cfg.word_ngrams})
    summ_path = cfg.metrics_dir / f"{run_name}_train_summary.json"
    if summ_path.exists():
        s = load_json(summ_path)
        m["lr"] = s.get("lr"); m["mean_epoch_time_sec"] = s.get("mean_epoch_time_sec")
        m["total_train_time_sec"] = s.get("total_train_time_sec")

    md = cfg.metrics_dir
    save_json(m, md / f"{run_name}_test_metrics.json")
    save_csv(m["per_class"], md / f"{run_name}_per_class.csv")
    (md / f"{run_name}_classification_report.txt").write_text(m["classification_report"])
    cm = np.array(m["confusion_matrix"])
    save_csv([{"true\\pred": c, **{p: int(v) for p, v in zip(cfg.class_names, row)}}
              for c, row in zip(cfg.class_names, cm)], md / f"{run_name}_confusion_matrix.csv")

    fd = cfg.figures_dir
    plot_confusion_matrix(cm, cfg.class_names, fd / f"{run_name}_confusion_matrix.png",
                          f"Confusion matrix (test) — {run_name}")
    plot_confusion_matrix(cm, cfg.class_names, fd / f"{run_name}_confusion_matrix_normalized.png",
                          f"Row-normalised confusion matrix — {run_name}", normalize=True)
    hist_path = md / f"{run_name}_history.json"
    if hist_path.exists():
        plot_training_curves(load_json(hist_path), fd, run_name)

    logger.info(f"[{run_name}] test accuracy {m['accuracy']:.4f} | macro-F1 {m['f1_macro']:.4f} | "
                f"test loss {test_loss:.4f}")
    logger.info("\n" + m["classification_report"])
    return m


# --------------------------------------------------------------------------- comparison
def build_comparison(cfg: Config) -> list:
    """Paper Table 1 vs. our test metrics (only values that exist are filled in)."""
    rows = []
    pairs = [("fastText, h=10", 1), ("fastText, h=10, bigram", 2)]
    official = cfg.metrics_dir / "official_fasttext_bigram_test_metrics.json"
    for paper_name, ng in pairs:
        run = default_run_name(get_config(word_ngrams=ng))
        path = cfg.metrics_dir / f"{run}_test_metrics.json"
        ours = load_json(path) if path.exists() else None
        row = {
            "model": paper_name,
            "paper_accuracy_pct": cfg.paper_results[paper_name],
            "reproduced_accuracy_pct": round(100 * ours["accuracy"], 2) if ours else "not run",
            "difference_pp": round(100 * ours["accuracy"] - cfg.paper_results[paper_name], 2) if ours else "",
            "reproduced_precision_macro_pct": round(100 * ours["precision_macro"], 2) if ours else "",
            "reproduced_recall_macro_pct": round(100 * ours["recall_macro"], 2) if ours else "",
            "reproduced_f1_macro_pct": round(100 * ours["f1_macro"], 2) if ours else "",
            "selected_lr": ours.get("lr") if ours else "",
            "mean_epoch_time_sec": round(ours["mean_epoch_time_sec"], 2) if ours and ours.get("mean_epoch_time_sec") else "",
        }
        if ng == 2 and official.exists():
            row["official_fasttext_lib_accuracy_pct"] = round(100 * load_json(official)["accuracy"], 2)
        rows.append(row)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))  # union, order preserved
    save_csv(rows, cfg.metrics_dir / "comparison.csv", fieldnames=fieldnames)
    save_json(rows, cfg.metrics_dir / "comparison.json")

    # Markdown table in the requested format
    lines = ["| Metric | Original Paper | Reproduced Results |", "|---|---|---|"]
    for r in rows:
        f2 = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else (v if v != "" else "not run")
        rep = f2(r["reproduced_accuracy_pct"])
        lines.append(f"| Test accuracy (%) — {r['model']} | {r['paper_accuracy_pct']} | {rep} |")
        for key, label in (("reproduced_precision_macro_pct", "Macro precision (%)"),
                           ("reproduced_recall_macro_pct", "Macro recall (%)"),
                           ("reproduced_f1_macro_pct", "Macro F1 (%)")):
            lines.append(f"| {label} — {r['model']} | not reported | {f2(r[key])} |")
        if r["mean_epoch_time_sec"] != "":
            lines.append(f"| Train time / epoch — {r['model']} | "
                         f"{'1s (bigram only)' if 'bigram' in r['model'] else 'not reported'} | {f2(r['mean_epoch_time_sec'])} s |")
        if "official_fasttext_lib_accuracy_pct" in r:
            lines.append(f"| Official fastText library, same split & machine — {r['model']} | — | "
                         f"{f2(r['official_fasttext_lib_accuracy_pct'])} |")
    md = "\n".join(lines) + "\n"
    (cfg.metrics_dir / "comparison.md").write_text(md)
    print(md)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained fastText models on the AG News test set")
    ap.add_argument("--word-ngrams", type=int, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--compare", action="store_true", help="only build the comparison table")
    a = ap.parse_args()
    cfg = get_config(word_ngrams=a.word_ngrams)
    if a.compare:
        build_comparison(cfg)
        return
    evaluate_run(cfg, a.run_name or default_run_name(cfg))


if __name__ == "__main__":
    main()
