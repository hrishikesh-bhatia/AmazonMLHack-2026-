import pandas as pd
import re
from collections import Counter

GT = "dataset/train/train_ground_truth.tsv"

print("Reading ground truth...")

gt = pd.read_csv(
    GT,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Rows:", len(gt))

# Number of matched IDs per S1
gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if not x else len(x.split(","))
)

print("\n===== MATCH COUNT DISTRIBUTION =====")
print(gt["match_count"].describe())

print("\n===== EXACT COUNTS =====")
print(gt["match_count"].value_counts().sort_index().head(30))

print("\nSingletons:")
print((gt["match_count"] == 0).sum())

print("Has match:")
print((gt["match_count"] > 0).sum())

print("\nTotal positive links:")
print(gt["match_count"].sum())

print("\nMaximum matches for one S1:")
print(gt["match_count"].max())


# Source distribution
s2 = 0
s3 = 0
both = 0

for x in gt["matched_entity_ids"]:
    if not x:
        continue

    ids = x.split(",")

    has2 = any(i.startswith("S2-") for i in ids)
    has3 = any(i.startswith("S3-") for i in ids)

    if has2:
        s2 += 1
    if has3:
        s3 += 1
    if has2 and has3:
        both += 1

print("\n===== SOURCE DISTRIBUTION =====")
print("S1s with S2 match:", s2)
print("S1s with S3 match:", s3)
print("S1s with BOTH:", both)