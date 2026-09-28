import pandas as pd
import os

files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]

for f in files:
    print("\n" + "=" * 70)
    print(f)
    print("Size:", round(os.path.getsize(f) / 1024**2, 2), "MB")

    df = pd.read_csv(f, sep="\t", nrows=5)
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))

    rows = sum(1 for _ in open(f, encoding="utf-8")) - 1
    print("Rows:", rows)