"""
Central configuration for the fastText (Joulin et al., 2016) reproduction on AG News.

Every hyperparameter lives here so it can be changed in one place. Values are
tagged in comments as:
    [PAPER]    stated in the paper (arXiv:1607.01759v3)
    [OFFICIAL] taken from the official script facebookresearch/fastText/classification-results.sh
    [ADAPTED]  our choice, because the PyTorch re-implementation or the environment differs

Any field can be overridden from the command line of src/train.py / src/evaluate.py
(see `--help`), or programmatically with `get_config(field=value, ...)`.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Repository root = parent of the `configs/` folder. All paths are relative to it,
# so the code works wherever the repository is cloned (e.g. /content/<repo> in Colab).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ------------------------------------------------------------------ general
    seed: int = 42                               # [ADAPTED] fixed seed for reproducibility
    num_threads: int = max(1, os.cpu_count() or 1)  # [ADAPTED] official uses -thread 4 (Hogwild)
    device: str = "cpu"                          # [ADAPTED] CPU only; fastText itself is CPU-only

    # ------------------------------------------------------------------ paths
    # These may be pointed at Google Drive (via environment variables) so that
    # checkpoints / processed data survive a Colab disconnect. See README.md.
    data_dir: Path = Path(os.environ.get("FT_DATA_DIR", PROJECT_ROOT / "data"))
    checkpoint_dir: Path = Path(os.environ.get("FT_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
    results_dir: Path = PROJECT_ROOT / "results"

    # ------------------------------------------------------------------ dataset
    dataset_name: str = "ag_news"
    # Mirrors of the original Zhang et al. (2015) AG News CSV files (identical bytes,
    # verified by SHA-256 below). Tried in order.
    csv_mirrors: Tuple[str, ...] = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{split}.csv",
    )
    # Original Google Drive archive used by the official fastText script (fallback).
    gdrive_archive_id: str = "0Bz8a_Dbh9QhbUDNpeUdjb0wxRms"
    sha256: Dict[str, str] = field(default_factory=lambda: {
        "train": "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e",
        "test": "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699",
    })
    expected_rows: Dict[str, int] = field(default_factory=lambda: {"train": 120000, "test": 7600})
    class_names: Tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")

    # Held-out validation split carved from the training set.
    # The paper selects the learning rate "on a validation set"; the official script
    # says the LR was "chosen by validation on a subset of the training set" but does
    # not give the split. 5% is our choice.
    val_fraction: float = 0.05                   # [ADAPTED]
    # If True, the final model is trained on all 120k training rows (as the official
    # script does) and validation curves are computed on the held-out split only for
    # monitoring (the split is then also seen in training, so curves are optimistic).
    # Default False: train on 95%, keep 5% truly held out.
    train_on_full_train: bool = False            # [ADAPTED]

    # ------------------------------------------------------------------ preprocessing
    lowercase: bool = True                       # [OFFICIAL] normalize_text: tr upper->lower
    fasttext_normalize: bool = True              # [OFFICIAL] same sed rules as normalize_text
    add_eos: bool = True                         # [OFFICIAL] fastText appends </s> to each line
    min_count: int = 1                           # [OFFICIAL] -minCount 1
    word_ngrams: int = 2                         # [PAPER]/[OFFICIAL] bigrams (-wordNgrams 2); 1 = unigrams
    bucket: int = 10_000_000                     # [PAPER]/[OFFICIAL] "10M bins if we only used bigrams"

    # ------------------------------------------------------------------ model
    embedding_dim: int = 10                      # [PAPER] h = 10 hidden units (-dim 10)
    # fastText initialises input vectors U(-1/dim, 1/dim) and the output matrix to zero.
    init_uniform_scale: float = 0.1              # [OFFICIAL] = 1/dim for dim=10
    zero_init_output: bool = True                # [OFFICIAL]
    # Loss: the paper describes a hierarchical softmax for many classes; the official
    # AG command does not pass -loss, so the default full softmax is used. With only
    # 4 classes they are computationally equivalent in practice.
    loss: str = "softmax"                        # [OFFICIAL]

    # ------------------------------------------------------------------ training
    epochs: int = 5                              # [PAPER] 5 epochs
    lr: float = 0.25                             # [OFFICIAL] LR used for ag_news
    lr_grid: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5)  # [PAPER] validation grid
    lr_schedule: str = "linear"                  # [OFFICIAL] linear decay to 0 over training
    # fastText does per-example SGD (Hogwild, 4 threads). We use small mini-batches with
    # a *summed* loss, so one step ~ batch_size consecutive per-example SGD steps.
    batch_size: int = 16                         # [ADAPTED]
    loss_reduction: str = "sum"                  # [ADAPTED] "sum" mimics per-example SGD
    shuffle_each_epoch: bool = True              # [ADAPTED] official shuffles the file once
    eval_batch_size: int = 1024                  # [ADAPTED]
    log_every_steps: int = 500                   # [ADAPTED]
    resume: bool = True                          # [ADAPTED] resume from checkpoints/<run>/last.pt
    save_every_epoch: bool = True                # [ADAPTED]

    # ------------------------------------------------------------------ reference results
    # Verified from Table 1 of arXiv:1607.01759v3 (AG test accuracy, %).
    paper_results: Dict[str, float] = field(default_factory=lambda: {
        "fastText, h=10": 91.5,
        "fastText, h=10, bigram": 92.5,
    })
    # Table 2 of the paper: AG training time per epoch for fastText h=10 bigram.
    paper_time_per_epoch: str = "1s (fastText, h=10, bigram; CPU with 20 threads, Section 3.1)"

    # ------------------------------------------------------------------ derived helpers
    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw" / "ag_news_csv"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"

    @property
    def figures_dir(self) -> Path:
        return Path(self.results_dir) / "figures"

    @property
    def metrics_dir(self) -> Path:
        return Path(self.results_dir) / "metrics"

    @property
    def logs_dir(self) -> Path:
        return Path(self.results_dir) / "logs"

    @property
    def effective_bucket(self) -> int:
        """fastText sets bucket=0 when wordNgrams<=1 and no char n-grams are used."""
        return self.bucket if self.word_ngrams > 1 else 0

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def get_config(**overrides) -> Config:
    """Return a Config with any given fields overridden (unknown keys raise)."""
    cfg = Config()
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config field: {key}")
        setattr(cfg, key, value)
    return cfg


def default_run_name(cfg: Config) -> str:
    tag = "bigram" if cfg.word_ngrams >= 2 else "unigram"
    return f"fasttext_h{cfg.embedding_dim}_{tag}"


__all__: List[str] = ["Config", "get_config", "default_run_name", "PROJECT_ROOT"]
