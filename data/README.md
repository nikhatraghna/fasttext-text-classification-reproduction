# Data

This folder is filled automatically by `python -m src.data_loader`; nothing here is committed
except this README (see `.gitignore`).

## AG News (Zhang, Zhao & LeCun, 2015)

| Split | Rows | Per class | SHA-256 |
|---|---|---|---|
| `raw/ag_news_csv/train.csv` | 120,000 | 30,000 | `76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e` |
| `raw/ag_news_csv/test.csv` | 7,600 | 1,900 | `521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699` |

Each row is `"class_index","title","description"` with classes
1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech.

These are the same files the official fastText script (`classification-results.sh`) downloads from
Google Drive. We download them from a byte-identical GitHub mirror and fall back to the Drive archive;
the SHA-256 values above are checked after every download.

## Derived files

* `processed/<settings>/` — cached vocabulary (`vocab.txt`) and encoded feature arrays
  (`train.npz`, `val.npz`, `test.npz`). The folder name encodes the preprocessing settings, so changing
  `word_ngrams`, `bucket`, `val_fraction`, ... automatically creates a new cache.
* `processed/official_fasttext/` — fastText-format text files for the optional official-library baseline.

## Validation split

AG News has no official validation set. A seeded random 5 % of the training file (6,000 documents,
`seed=42`) is held out for learning-rate selection and validation curves (`configs/config.py:
val_fraction`, `train_on_full_train`).
