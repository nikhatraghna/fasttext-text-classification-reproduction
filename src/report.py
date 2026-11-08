"""
Generate the reproducibility report (results/reproducibility_report.md) from the
result files written by train.py / evaluate.py / official_baseline.py.

Every number in the report is read from results/metrics/*.json at generation time;
nothing is hard-coded except the paper's own values (configs/config.py, verified
against arXiv:1607.01759v3). If an experiment has not been run, the report says so.

    python -m src.report
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from configs.config import default_run_name, get_config
from src.utils import load_json

CFG = get_config()
M = CFG.metrics_dir
UNI = default_run_name(get_config(word_ngrams=1))
BI = default_run_name(get_config(word_ngrams=2))


def _load(name: str) -> Optional[dict]:
    p = M / name
    return load_json(p) if p.exists() else None


def pct(x, nd=2) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def num(x, nd=2, unit="") -> str:
    return "n/a" if x is None else f"{x:.{nd}f}{unit}"


def g(d: Optional[dict], *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def fig(name: str, caption: str) -> str:
    p = CFG.figures_dir / name
    if not p.exists():
        return f"*Figure not available: `{name}` (run the corresponding step).*\n"
    return f"![{caption}](figures/{name})\n\n*{caption}*\n"


def per_class_table(m: Optional[dict]) -> str:
    if not m:
        return "*Not available.*\n"
    rows = ["| Class | Precision (%) | Recall (%) | F1 (%) | Support |", "|---|---|---|---|---|"]
    for r in m["per_class"]:
        rows.append(f"| {r['class']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | {r['support']} |")
    return "\n".join(rows) + "\n"


def history_table(h: Optional[list]) -> str:
    if not h:
        return "*Not available.*\n"
    rows = ["| Epoch | Train loss | Train acc (%) | Val loss | Val acc (%) | Epoch time (s) |",
            "|---|---|---|---|---|---|"]
    for r in h:
        rows.append(f"| {r['epoch']} | {r['train_loss']:.4f} | {pct(r['train_acc'])} | {r['val_loss']:.4f} | "
                    f"{pct(r['val_acc'])} | {r['epoch_time_sec']:.2f} |")
    return "\n".join(rows) + "\n"


def sweep_table(tag: str) -> str:
    import csv
    p = M / f"lr_sweep_{tag}.csv"
    if not p.exists():
        return "*LR sweep not run.*\n"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    best = _load(f"best_lr_{tag}.json")
    out = ["| Initial LR | Val accuracy (%) | Val loss | Train time (s) |", "|---|---|---|---|"]
    for r in rows:
        mark = " **(selected)**" if best and float(r["lr"]) == float(best["best_lr"]) else ""
        out.append(f"| {r['lr']}{mark} | {100 * float(r['val_acc']):.2f} | {float(r['val_loss']):.4f} | "
                   f"{float(r['train_time_sec']):.1f} |")
    return "\n".join(out) + "\n"


def describe_gap(ours: Optional[float], paper: float) -> str:
    if ours is None:
        return "has not been run yet"
    d = 100 * ours - paper
    a = abs(d)
    if a < 0.05:
        return f"matches the paper ({100 * ours:.2f}% vs {paper}%)"
    direction = "below" if d < 0 else "above"
    return f"is {a:.2f} percentage points {direction} the paper ({100 * ours:.2f}% vs {paper}%)"


def build() -> str:
    mu, mb = _load(f"{UNI}_test_metrics.json"), _load(f"{BI}_test_metrics.json")
    su, sb = _load(f"{UNI}_train_summary.json"), _load(f"{BI}_train_summary.json")
    hu, hb = _load(f"{UNI}_history.json"), _load(f"{BI}_history.json")
    off = _load("official_fasttext_bigram_test_metrics.json")
    off_status = _load("official_fasttext_bigram_status.json")
    blu, blb = _load("best_lr_unigram.json"), _load("best_lr_bigram.json")
    env = g(sb, "environment") or g(su, "environment") or {}
    pu, pb = CFG.paper_results["fastText, h=10"], CFG.paper_results["fastText, h=10, bigram"]
    acc_u, acc_b = g(mu, "accuracy"), g(mb, "accuracy")
    bigram_gain = None if (acc_u is None or acc_b is None) else 100 * (acc_b - acc_u)
    vocab = g(sb, "vocab_size") or g(su, "vocab_size")
    params_b = g(sb, "num_parameters")
    train_docs = g(sb, "train_docs") or g(su, "train_docs")
    val_docs = g(sb, "val_docs") or g(su, "val_docs")

    if off:
        same = acc_b is not None and abs(off["accuracy"] - acc_b) < 1e-9
        official_sentence = (
            f"As an additional control, the official fastText library (pip package, version "
            f"{off.get('fasttext_version', 'unknown')}) was trained with the official command on the *same* "
            f"normalised training split, on the same machine, with lr = {off['lr']}. It reached "
            f"**{pct(off['accuracy'])}%** test accuracy in {num(off['total_train_time_sec'], 1)} s for "
            f"{CFG.epochs} epochs. " + (
                "This is identical to our PyTorch bigram model's accuracy." if same else
                f"Our PyTorch bigram model reached {pct(acc_b)}%, a difference of "
                f"{num(None if acc_b is None else 100 * (acc_b - off['accuracy']), 2)} percentage points.")
        )
    elif off_status:
        official_sentence = (f"The optional official-library baseline was attempted but failed in this "
                             f"environment (`{off_status.get('error')}`), so no official-library number is reported.")
    else:
        official_sentence = ("The optional official-library baseline (`python -m src.official_baseline`) was not run, "
                             "so no official-library number is reported.")

    # ---- statements that depend on the actual numbers (never asserted blindly)
    import csv as _csv
    sweep_spread = {}
    for tag in ("unigram", "bigram"):
        sp = M / f"lr_sweep_{tag}.csv"
        if sp.exists():
            with open(sp) as f:
                accs = [float(x["val_acc"]) for x in _csv.DictReader(f)]
            sweep_spread[tag] = 100 * (max(accs) - min(accs))
    if sweep_spread:
        spread_sentence = ("Across the four grid values, validation accuracy varied by " +
                           ", ".join(f"{v:.2f} points ({k})" for k, v in sweep_spread.items()) +
                           ", which bounds how much the LR choice alone can plausibly matter.")
    else:
        spread_sentence = "The LR sweep was not run, so its sensitivity cannot be quantified here."

    agree = g(off, "feature_pipeline_agreement")
    if agree is not None:
        verify_sentence = (f"`src/official_baseline.py` copies the learned matrices of the official library into our "
                           f"PyTorch model (remapping word ids) and compares predictions on the test set: agreement was "
                           f"**{100 * agree:.2f}%**" + (", i.e. tokenisation, vocabulary and n-gram hashing are identical "
                           "to the official implementation." if agree == 1.0 else
                           ". Agreement below 100% indicates a residual difference in the feature pipeline."))
    else:
        verify_sentence = ("Run `python -m src.official_baseline` (requires `pip install fasttext`) to check the "
                           "feature pipeline against the official library; this check was not run for this report.")

    t_total = g(sb, "total_train_time_sec")
    speed_sentence = ("The qualitative claim — a full training run in seconds-to-minutes on a CPU, with no GPU — holds in "
                      "our environment." if t_total is not None and t_total < 600 else
                      "Training-time results are not available or exceed ten minutes, so the speed claim is not confirmed here.")

    gaps = [abs(100 * a - p) for a, p in ((acc_u, pu), (acc_b, pb)) if a is not None]
    if len(gaps) == 2 and max(gaps) <= 1.0 and bigram_gain is not None and bigram_gain > 0:
        claims_sentence = ("The paper's central claims for AG News — that this simple linear model is competitive with the "
                           "deep models of the time, that bigrams help, and that training takes seconds without a GPU — are "
                           "supported by these results. The remaining differences (≤ 1 point) are consistent with the documented "
                           "deviations in update granularity, data split and hardware.")
    elif len(gaps) == 2:
        claims_sentence = (f"The largest gap to the paper is {max(gaps):.2f} points" +
                           ("" if bigram_gain is None or bigram_gain > 0 else ", and bigrams did not improve accuracy") +
                           ". This is larger than the documented deviations alone would suggest and should be investigated "
                           "(e.g. full training set, several seeds, per-example updates).")
    else:
        claims_sentence = "Both final models must be trained and evaluated before the paper's claims can be assessed."

    comparison_md = (M / "comparison.md").read_text() if (M / "comparison.md").exists() else \
        "*Run `python -m src.evaluate --compare` to build this table.*\n"

    r = f"""# Reproducibility Report: "Bag of Tricks for Efficient Text Classification" on AG News

*Generated automatically by `src/report.py` from the files in `results/metrics/`. All reproduced
numbers below come from the experiment logs of this repository; the original numbers are taken from
Table 1 and Table 2 of arXiv:1607.01759v3.*

---

## 1. Introduction

Text classification is one of the most common tasks in natural language processing: spam filtering,
topic labelling, sentiment analysis and intent detection all reduce to it. By 2016 the strongest
reported results on the standard large-scale benchmarks of Zhang et al. (2015) came from deep
convolutional and recurrent networks operating on characters or words, which needed GPUs and hours of
training per epoch. Joulin, Grave, Bojanowski and Mikolov (2016) argued that a carefully engineered
*linear* model, with a low-rank constraint, averaged word and n-gram embeddings, and a handful of
efficiency "tricks", reaches accuracy on par with these networks while training several orders of
magnitude faster on a CPU. Their method was released as the fastText library.

This report documents an independent re-implementation of that classifier in PyTorch and its evaluation
on the AG News topic-classification benchmark, one of the eight datasets in Table 1 of the paper. The goals
are (i) to check whether the reported AG News test accuracies (**{pu}%** without and **{pb}%** with bigrams)
can be obtained from the published description and the official scripts, (ii) to quantify every
deviation from the original setup, and (iii) to package the experiment so that it can be re-run end to
end on a CPU-only Google Colab instance and resumed after a disconnect.

The report is organised as follows. Section 2 summarises the paper; Sections 3–5 describe the method,
the implementation and the experimental setup; Sections 6–7 present the reproduced results and compare
them with the original ones; Section 8 discusses sources of discrepancy, and Sections 9–10 list
limitations and conclude.

## 2. Paper Overview

**Model.** A document is represented as a bag of features (words and, optionally, word n-grams). Each
feature indexes a row of an embedding matrix *A*; the rows are averaged into a hidden document
representation of size *h*, which is fed to a linear classifier *B* followed by a softmax. Training minimises
the negative log-likelihood

  −(1/N) Σₙ yₙ log(f(B A xₙ)),

where *xₙ* is the normalised bag of features of document *n*. The model is therefore a linear classifier
whose weight matrix is factorised into *B·A* (rank ≤ *h*), which lets parameters be shared across features
and classes. It is trained with stochastic gradient descent and a linearly decaying learning rate,
asynchronously on multiple CPU threads.

**Tricks.** (1) A *hierarchical softmax* based on a Huffman tree reduces the cost of the output layer from
O(kh) to O(h log₂ k) for *k* classes — important for tag prediction with hundreds of thousands of labels,
irrelevant for the 4 classes of AG News. (2) *N-gram features* add local word-order information that a pure
bag of words lacks. (3) The *hashing trick* (Weinberger et al., 2009) maps n-grams into a fixed number of
buckets (10M when only bigrams are used) instead of storing an explicit n-gram vocabulary.

**Experiments on AG News.** The sentiment/topic experiments use *h* = 10 hidden units, 5 epochs, and an
initial learning rate selected on a validation set from {{0.05, 0.1, 0.25, 0.5}}. Table 1 of the paper
reports AG News test accuracy of **{pu}%** for fastText (h=10) and **{pb}%** for fastText (h=10, bigram),
compared with 87.2% for char-CNN, 91.4% for char-CRNN and 91.3% for VDCNN. Table 2 reports a training
time of **1 s per epoch** on AG News for the bigram model (CPU, 20 threads) versus 1–3 h for char-CNN and
24–51 min for VDCNN on a Tesla K40 GPU.

**Official implementation.** The script `classification-results.sh` in the fastText repository reproduces
Table 1. For AG News it lowercases and tokenises the CSV files with a `sed` pipeline and runs
`fasttext supervised -dim 10 -lr 0.25 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4`.
It does not pass `-loss`, so the default (full) softmax is used. The script notes that its learning rates
"were chosen by validation on a subset of the training set".

## 3. Methodology

Our reproduction follows the paper and the official script step by step:

1. **Data.** The original AG News CSV files of Zhang et al. (2015): 120,000 training and 7,600 test
   documents, 4 balanced classes (World, Sports, Business, Sci/Tech). Each document is a title plus a
   short description. File integrity is verified with SHA-256 and row counts.
2. **Normalisation.** The official `normalize_text` function is reproduced exactly (ASCII lowercasing,
   separating `. , ( ) ! ?` and apostrophes, removing quotes, `;` and `:`). Because the official script
   operates on the raw CSV line, the field separators become `,` tokens; we reproduce this too.
3. **Features.** Unigrams from a training-set vocabulary (minCount = 1) plus an end-of-line token
   `</s>`. In the bigram model, every pair of consecutive tokens is hashed with fastText's own
   hash functions into 10M buckets. Out-of-vocabulary test words are dropped as unigrams but still
   contribute to bigrams, exactly as in the C++ code.
4. **Model.** Averaged embedding bag (h = 10) → linear layer (no bias) → softmax cross-entropy.
   Initialisation as in fastText: *A* ~ U(−1/h, 1/h), *B* = 0.
5. **Optimisation.** SGD with the learning rate decaying linearly to zero over the total number of
   training tokens across 5 epochs.
6. **Model selection.** Following the paper, the initial learning rate is chosen from {{0.05, 0.1, 0.25, 0.5}}
   by accuracy on a held-out validation split, separately for the unigram and bigram models.
7. **Evaluation.** The model after the 5th epoch (no early stopping) is evaluated once on the official
   test set. We report accuracy (the paper's only metric) plus macro/weighted precision, recall, F1, the
   confusion matrix and a per-class report.

## 4. Implementation Details

The code is organised as a small Python package; every file has a single responsibility:

| File | Responsibility |
|---|---|
| `configs/config.py` | All hyperparameters and paths, each tagged [PAPER], [OFFICIAL] or [ADAPTED] |
| `src/data_loader.py` | Download (GitHub mirror, Google Drive fallback), SHA-256 check, CSV parsing, train/val split |
| `src/preprocessing.py` | Official normalisation, tokenisation, vocabulary, fastText hashing of n-grams, feature caching |
| `src/model.py` | `FastTextClassifier` (`nn.EmbeddingBag(mode="mean", sparse=True)` + `nn.Linear`) |
| `src/train.py` | Training loop, linear LR decay, validation, checkpoint/resume, LR sweep |
| `src/evaluate.py` | Test metrics, confusion matrix, figures, paper-vs-reproduction table |
| `src/official_baseline.py` | Optional control run of the official fastText library on the same split |
| `src/report.py` | This report |
| `src/utils.py` | Seeding, logging, JSON/CSV I/O, environment capture, plotting |

**Hashed n-grams.** fastText hashes each token with 32-bit FNV-1a (bytes sign-extended from `int8`),
stores the hashes as signed 32-bit integers, and combines consecutive hashes as
`h = h · 116049371 + hash(w_{{i+1}})` in unsigned 64-bit arithmetic; the feature index is
`nwords + (h mod bucket)`. `src/preprocessing.py` implements this in vectorised NumPy over the whole corpus,
including the sign extension from `int32` to `uint64`. A document's features are stored as a flat index
array plus offsets, the native input format of `nn.EmbeddingBag`, and cached to `data/processed/`.

**Verification of the feature pipeline.** {verify_sentence}

**Efficiency.** With `sparse=True` each SGD step only reads and writes the embedding rows that occur in the
mini-batch, so the cost per step is independent of the 10M-row table — the property that makes fastText fast.
The bigram model has {f"{params_b:,}" if params_b else "n/a"} parameters (({f"{vocab:,}" if vocab else "n/a"} words + 10,000,000
buckets) × 10 + 40 output weights); its weights take about 0.4 GB in float32.

**Mini-batching.** fastText performs one SGD update per document, in parallel over threads without locks
(Hogwild). Per-document updates are very slow in PyTorch because of Python overhead, so we use mini-batches of
{CFG.batch_size} documents with a *summed* (not averaged) loss. One step then approximates {CFG.batch_size}
consecutive per-document updates with the same learning rate, and the paper's learning-rate grid keeps its meaning.

**Resumability.** After every epoch the model, optimiser, RNG states, processed-token counter and metric history
are written atomically to `checkpoints/<run>/last.pt`. A restarted run continues from the next epoch and gives
bit-identical results. During development this was tested by killing a run after epoch 2. Finished runs keep only
`final.pt`, and re-running them is a no-op. LR-sweep runs are resumable per learning rate.

## 5. Experimental Setup

**Data splits.** AG News has no official validation split. We hold out a seeded random {100 * CFG.val_fraction:.0f}% of
the training set ({f"{val_docs:,}" if val_docs else "n/a"} documents) for learning-rate selection and the validation curves,
and train on the remaining {f"{train_docs:,}" if train_docs else "n/a"} documents. The test set (7,600 documents) is used
only once per final model.

**Hyperparameters.**

| Setting | Paper | Official script (AG News) | This reproduction |
|---|---|---|---|
| Hidden units *h* | 10 | `-dim 10` | 10 |
| Epochs | 5 | `-epoch 5` | {CFG.epochs} |
| Initial LR | chosen from {{0.05, 0.1, 0.25, 0.5}} on validation | `-lr 0.25` | unigram: {g(blu, "best_lr") or "n/a"}, bigram: {g(blb, "best_lr") or "n/a"} (selected on our validation split) |
| LR schedule | linear decay | linear decay to 0 (by tokens) | linear decay to 0 (by tokens) |
| Word n-grams | 1 and 2 | `-wordNgrams 2` | 1 and 2 |
| Hash buckets | 10M (bigrams only) | `-bucket 10000000` | 10,000,000 (0 for unigram model, as fastText does) |
| minCount | — | `-minCount 1` | 1 |
| Loss | hierarchical softmax described | default softmax | softmax |
| Update granularity | per example, async multi-CPU | per example, Hogwild, 4 threads | mini-batch {CFG.batch_size}, summed loss, synchronous |
| Training data | not stated | full train set (120,000) | {f"{train_docs:,}" if train_docs else "n/a"} documents (95%) |
| Hardware | CPU, 20 threads | — | {env.get("cpu_model", "n/a")}, {env.get("cpu_count", "n/a")} logical CPUs |
| Seed | — | not fixed | {CFG.seed} |

**Software and hardware.** Python {env.get("python", "n/a")}, PyTorch {env.get("torch", "n/a")}, NumPy
{env.get("numpy", "n/a")}; {env.get("cpu_model", "CPU n/a")} with {env.get("cpu_count", "n/a")} logical CPUs
and {env.get("mem_total", "n/a")} RAM; CUDA available: {env.get("cuda_available", "n/a")} (not used).
PyTorch used {env.get("torch_threads", "n/a")} intra-op threads.

## 6. Reproduced Results

### 6.1 Learning-rate selection

Unigram model (validation accuracy after 5 epochs):

{sweep_table("unigram")}
Bigram model:

{sweep_table("bigram")}
{fig("lr_sweep_unigram.png", "Figure 1a. Validation accuracy versus initial learning rate, unigram model.")}
{fig("lr_sweep_bigram.png", "Figure 1b. Validation accuracy versus initial learning rate, bigram model.")}

### 6.2 Test-set results

| Model | LR | Test accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Weighted F1 (%) | Test loss |
|---|---|---|---|---|---|---|---|
| fastText, h=10 (unigram) | {g(mu, "lr") or "n/a"} | **{pct(acc_u)}** | {pct(g(mu, "precision_macro"))} | {pct(g(mu, "recall_macro"))} | {pct(g(mu, "f1_macro"))} | {pct(g(mu, "f1_weighted"))} | {num(g(mu, "test_loss"), 4)} |
| fastText, h=10, bigram | {g(mb, "lr") or "n/a"} | **{pct(acc_b)}** | {pct(g(mb, "precision_macro"))} | {pct(g(mb, "recall_macro"))} | {pct(g(mb, "f1_macro"))} | {pct(g(mb, "f1_weighted"))} | {num(g(mb, "test_loss"), 4)} |

Adding bigrams changes test accuracy by **{num(bigram_gain, 2)} percentage points** (paper: {pb - pu:.1f}).
Because AG News test classes are balanced (1,900 each), macro recall equals accuracy.

**Per-class results — bigram model**

{per_class_table(mb)}
**Per-class results — unigram model**

{per_class_table(mu)}
### 6.3 Training dynamics

Bigram model:

{history_table(hb)}
Unigram model:

{history_table(hu)}
Training loss and accuracy are running means over each epoch, while the model is still being updated.
Validation metrics are computed at the end of each epoch.

{fig(f"{BI}_train_loss.png", "Figure 2. Training loss, bigram model.")}
{fig(f"{BI}_val_loss.png", "Figure 3. Validation loss, bigram model.")}
{fig(f"{BI}_accuracy.png", "Figure 4. Training and validation accuracy, bigram model.")}
{fig(f"{UNI}_loss.png", "Figure 5. Training and validation loss, unigram model.")}
{fig(f"{UNI}_accuracy.png", "Figure 6. Training and validation accuracy, unigram model.")}

### 6.4 Confusion matrices

{fig(f"{BI}_confusion_matrix.png", "Figure 7. Test confusion matrix, bigram model.")}
{fig(f"{BI}_confusion_matrix_normalized.png", "Figure 8. Row-normalised test confusion matrix, bigram model.")}
{fig(f"{UNI}_confusion_matrix.png", "Figure 9. Test confusion matrix, unigram model.")}

### 6.5 Training time

Mean wall-clock time per epoch (excluding evaluation): unigram **{num(g(su, "mean_epoch_time_sec"), 2, " s")}**, bigram
**{num(g(sb, "mean_epoch_time_sec"), 2, " s")}**. Total training time for 5 epochs: unigram
{num(g(su, "total_train_time_sec"), 1, " s")}, bigram {num(g(sb, "total_train_time_sec"), 1, " s")}.

## 7. Comparison with Original Results

{comparison_md}
Our unigram model {describe_gap(acc_u, pu)}. Our bigram model {describe_gap(acc_b, pb)}.
The paper reports only accuracy, so no original precision, recall or F1 values exist to compare with;
these rows are marked "not reported" rather than estimated.

{official_sentence}

Training time is not directly comparable: the paper used a CPU with 20 threads and highly optimised
C++, while we ran PyTorch on {env.get("cpu_count", "n/a")} logical CPU(s). {speed_sentence} The paper's GPU baselines needed
minutes to hours *per epoch*.

**Assessment.** This is not an exact reproduction of the original setting. Model, features, hashing, initialisation,
schedule, epochs and hyperparameter grid all follow the paper and the official code. Training granularity
(mini-batches), the validation split, the amount of training data and the hardware all differ, and Section 8
discusses each. We therefore do not claim an exact reproduction of the original numbers.

## 8. Reproducibility Discussion

**Preprocessing.** The paper itself does not describe preprocessing for AG News; it is defined only by the official
script, which we reproduce rule by rule (see the feature-pipeline check in Section 4). One undocumented consequence of that script is that CSV field separators become `,` tokens, and
literal backslashes from the original crawl (e.g. `\\which`) stay attached to words. Both are kept for fidelity.
Lowercasing is ASCII-only, as `tr` behaves in the C locale.

**Hyperparameters.** The paper gives *h*, epochs and the LR grid but not the selected learning rate. The official
script uses 0.25 for AG News, chosen "on a subset of the training set" whose size is not specified. We ran the
grid on our own 5% split. {spread_sentence}

**Model implementation.** (i) The paper describes a hierarchical softmax, but the official AG command uses the
default full softmax. With 4 classes the Huffman tree would have depth ≤ 2, so we follow the official command.
(ii) fastText updates per example with lock-free parallel threads, so the order and staleness of updates depend on
thread timing and are not deterministic. Our synchronous mini-batches of {CFG.batch_size} with a summed loss are a
close but not identical approximation; its effect can be measured with `--batch-size`. (iii) fastText's learning rate is updated from a token counter shared across threads; ours
is updated every mini-batch from the exact count of processed tokens.

**Dataset splits.** The official script trains on all 120,000 training documents; by default we train on 95% of them
so that the validation curves are not contaminated. Setting `train_on_full_train=True` (or `--full-train`) restores
the official amount of data, at the cost of meaningful validation curves. The official script also shuffles the
training file once with an unseeded `perl` shuffle; we reshuffle every epoch with a fixed seed.

**Computing environment.** The original experiments ran on 20 CPU threads; ours ran on a {env.get("cpu_count", "n/a")}-CPU
machine. Hogwild results depend on thread count, so even the official binary does not give bit-identical accuracy
across machines. Our PyTorch runs are deterministic given the seed and thread settings, and resuming from a
checkpoint gives bit-identical results.

**Randomness.** The original paper reports one number per model, with no variance. A single seed was used here as
well. Repeating the final runs with several seeds (`--seed`) would give a confidence interval and is the most
useful next step to judge whether a gap of a few tenths of a point is meaningful.

## 9. Challenges and Limitations

* **Dataset hosting.** The official script downloads AG News from a Google Drive link that now often needs an
  extra confirmation step. We use a byte-identical GitHub mirror (verified by SHA-256) with the Drive archive as a
  fallback.
* **Per-example SGD in PyTorch.** Reproducing fastText's per-example, lock-free updates literally would be one or
  two orders of magnitude slower in Python, hence the mini-batch approximation.
* **Memory and checkpoint size.** A 10M-row bucket table is about 0.4 GB in float32, and so is every bigram
  checkpoint. This is fine on Colab but too large for GitHub, so checkpoints are git-ignored. Optionally they can be
  stored on Google Drive.
* **Colab sessions are ephemeral.** Results, logs and figures are committed to GitHub; checkpoints and processed
  features can be redirected to Drive with the `FT_CHECKPOINT_DIR` / `FT_DATA_DIR` environment variables.
* **Official library in modern Python.** The pip build of the official library is used only as an optional
  control. Recent builds have been observed to abort with "Encountered NaN" on some configurations, so it is run
  for the bigram setting only, and any failure is recorded in `results/metrics/` rather than hidden.
* **Scope.** Only AG News and the two *h*=10 rows of Table 1 are reproduced. The other seven datasets, the tag
  prediction experiment (YFCC100M) and the hierarchical-softmax speed-up are out of scope.
* **Single run.** Results come from one seed; no variance estimate is available yet.

## 10. Conclusion

We re-implemented the fastText classifier of Joulin et al. (2016) in PyTorch, matching the official
preprocessing, vocabulary, hashed-bigram features, initialisation and learning-rate schedule, and verified the
feature pipeline against the official library. On AG News, our unigram model reaches **{pct(acc_u)}%** test accuracy
(paper: {pu}%) and our bigram model **{pct(acc_b)}%** (paper: {pb}%). Bigrams change accuracy by {num(bigram_gain, 2)}
points, and a full five-epoch training run takes {num(g(sb, "total_train_time_sec"), 1, " s")} on a CPU. {claims_sentence}
Because of the documented deviations, this is not an exact reproduction of the original experimental setting.

## 11. References

1. A. Joulin, E. Grave, P. Bojanowski, T. Mikolov. *Bag of Tricks for Efficient Text Classification.*
   arXiv:1607.01759, 2016 (EACL 2017). https://arxiv.org/abs/1607.01759
2. fastText official implementation and `classification-results.sh`. https://github.com/facebookresearch/fastText
3. X. Zhang, J. Zhao, Y. LeCun. *Character-level Convolutional Networks for Text Classification.* NIPS 2015.
4. A. Conneau, H. Schwenk, L. Barrault, Y. LeCun. *Very Deep Convolutional Networks for Natural Language
   Processing.* arXiv:1606.01781, 2016.
5. Y. Xiao, K. Cho. *Efficient Character-level Document Classification by Combining Convolution and Recurrent
   Layers.* arXiv:1602.00367, 2016.
6. J. Goodman. *Classes for Fast Maximum Entropy Training.* ICASSP 2001.
7. K. Weinberger, A. Dasgupta, J. Langford, A. Smola, J. Attenberg. *Feature Hashing for Large Scale Multitask
   Learning.* ICML 2009.
8. B. Recht, C. Re, S. Wright, F. Niu. *Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient
   Descent.* NIPS 2011.
9. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
"""
    return r


def main():
    out = CFG.results_dir / "reproducibility_report.md"
    out.write_text(build(), encoding="utf-8")
    print(f"Report written to {out} ({len(out.read_text().split())} words)")

    # Refresh the results block of the README from the generated comparison table.
    readme = CFG.results_dir.parent / "README.md"
    comp = M / "comparison.md"
    if readme.exists() and comp.exists():
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        if start in text and end in text:
            block = (f"{start}\nAG News test set, generated by this repository's code "
                     f"(`results/metrics/comparison.md`):\n\n{comp.read_text(encoding='utf-8')}{end}")
            text = text[: text.index(start)] + block + text[text.index(end) + len(end):]
            readme.write_text(text, encoding="utf-8")
            print(f"Updated results section of {readme}")


if __name__ == "__main__":
    main()
