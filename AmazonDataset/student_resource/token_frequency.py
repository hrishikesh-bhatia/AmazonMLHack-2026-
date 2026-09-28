import pandas as pd
import re
import unicodedata
from collections import Counter
import pickle
import os
import time

TRAIN = "dataset/train"

FILES = [
    f"{TRAIN}/train_source2.tsv",
    f"{TRAIN}/train_source3.tsv",
]


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = unicodedata.normalize("NFKC", x)

    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def get_tokens(x):
    text = normalize_text(x)

    # Ignore extremely short tokens
    return set(
        t for t in text.split()
        if len(t) >= 2
    )


counter = Counter()

start = time.time()

for file in FILES:

    print("\nScanning:", file)

    rows = 0

    for chunk in pd.read_csv(
        file,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            # IMPORTANT:
            # Count document frequency, not total occurrences.
            tokens = (
                get_tokens(row.business_name)
                |
                get_tokens(row.business_address)
            )

            for token in tokens:
                counter[token] += 1

        rows += len(chunk)

        if rows % 1_000_000 == 0:
            print(
                f"  processed {rows:,} rows | "
                f"unique tokens={len(counter):,}"
            )

    print("Finished rows:", rows)


print("\n================================")
print("TOKEN FREQUENCY ANALYSIS")
print("================================")

print("Unique tokens:", len(counter))

print("\nMost common tokens:")

for token, freq in counter.most_common(50):
    print(f"{token:30s} {freq:,}")


print("\nFrequency buckets:")

buckets = [
    (1, 1),
    (2, 5),
    (6, 10),
    (11, 50),
    (51, 100),
    (101, 500),
    (501, 1000),
    (1001, 5000),
    (5001, 10000),
    (10001, 50000),
    (50001, 100000),
    (100001, 10_000_000),
]

for lo, hi in buckets:

    count = sum(
        lo <= f <= hi
        for f in counter.values()
    )

    print(
        f"{lo:>7,} - {hi:>10,}: "
        f"{count:,} tokens"
    )


with open("token_df.pkl", "wb") as f:
    pickle.dump(counter, f)

print("\nSaved token_df.pkl")

print(
    "\nTime:",
    round(time.time() - start, 2),
    "seconds"
)