import pandas as pd
import re
import unicodedata
from collections import Counter

TRAIN = "dataset/train"

GT_FILE = f"{TRAIN}/train_ground_truth.tsv"

SAMPLE_SIZE = 500_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = unicodedata.normalize("NFKC", x)

    # Keep unicode letters/numbers
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def tokens(x):
    x = normalize_text(x)

    # Remove extremely short tokens
    return set(t for t in x.split() if len(t) >= 2)


def jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def overlap(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

positive_links = []

for row in gt.itertuples(index=False):

    if not row.matched_entity_ids:
        continue

    for mid in row.matched_entity_ids.split(","):
        positive_links.append(
            (row.source1_entity_id, mid)
        )

positive_links = positive_links[:SAMPLE_SIZE]

print("Positive sample:", len(positive_links))


needed_s1 = set()
needed_other = set()

for s1, mid in positive_links:
    needed_s1.add(s1)
    needed_other.add(mid)


# ============================================================
# LOAD RECORDS
# ============================================================

print("Loading S1...")

s1_data = {}

for chunk in pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    dtype=str,
    chunksize=200_000,
    keep_default_na=False
):

    mask = chunk["entity_id"].isin(needed_s1)

    for row in chunk.loc[mask].itertuples(index=False):

        s1_data[row.entity_id] = {
            "name": tokens(row.business_name),
            "address": tokens(row.business_address)
        }

    if len(s1_data) == len(needed_s1):
        break


print("S1 loaded:", len(s1_data))


print("Loading S2/S3...")

other_data = {}

for filename in [
    f"{TRAIN}/train_source2.tsv",
    f"{TRAIN}/train_source3.tsv"
]:

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=200_000,
        keep_default_na=False
    ):

        mask = chunk["entity_id"].isin(needed_other)

        for row in chunk.loc[mask].itertuples(index=False):

            other_data[row.entity_id] = {
                "name": tokens(row.business_name),
                "address": tokens(row.business_address)
            }

        if len(other_data) == len(needed_other):
            break


print("Other loaded:", len(other_data))


# ============================================================
# ANALYZE
# ============================================================

name_jaccards = []
name_overlaps = []

address_jaccards = []
address_overlaps = []

name_or_address = []

for s1_id, other_id in positive_links:

    a = s1_data[s1_id]
    b = other_data[other_id]

    nj = jaccard(a["name"], b["name"])
    no = overlap(a["name"], b["name"])

    aj = jaccard(a["address"], b["address"])
    ao = overlap(a["address"], b["address"])

    name_jaccards.append(nj)
    name_overlaps.append(no)

    address_jaccards.append(aj)
    address_overlaps.append(ao)


# ============================================================
# RESULTS
# ============================================================

def pct(values, threshold):
    return sum(x >= threshold for x in values) / len(values) * 100


print("\n========================================")
print("NAME TOKEN JACCARD")
print("========================================")

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f">= {t:.1f}: {pct(name_jaccards, t):.2f}%")


print("\n========================================")
print("NAME TOKEN OVERLAP")
print("========================================")

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f">= {t:.1f}: {pct(name_overlaps, t):.2f}%")


print("\n========================================")
print("ADDRESS TOKEN JACCARD")
print("========================================")

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f">= {t:.1f}: {pct(address_jaccards, t):.2f}%")


print("\n========================================")
print("ADDRESS TOKEN OVERLAP")
print("========================================")

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f">= {t:.1f}: {pct(address_overlaps, t):.2f}%")


print("\n========================================")
print("COMBINED NAME/ADDRESS")
print("========================================")

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:

    recovered = 0

    for nj, aj in zip(name_jaccards, address_jaccards):

        if nj >= t or aj >= t:
            recovered += 1

    print(
        f"threshold {t:.1f}: "
        f"{recovered / len(name_jaccards) * 100:.2f}%"
    )