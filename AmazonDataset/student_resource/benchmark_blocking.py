import pandas as pd
import re
import unicodedata
from collections import defaultdict
import time

TRAIN = "dataset/train"

SOURCES = [
    f"{TRAIN}/train_source2.tsv",
    f"{TRAIN}/train_source3.tsv",
]

GT_FILE = f"{TRAIN}/train_ground_truth.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    # Unicode normalization
    x = unicodedata.normalize("NFKC", x)

    # Replace punctuation with spaces.
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)

    # Collapse whitespace
    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_name(x):
    return normalize_text(x)


def normalize_address(x):
    return normalize_text(x)


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

print("Ground truth rows:", len(gt))


# ============================================================
# Extract true links
# ============================================================

print("Extracting positive links...")

positive_links = []

for row in gt.itertuples(index=False):
    s1 = row.source1_entity_id
    matches = row.matched_entity_ids

    if not matches:
        continue

    for mid in matches.split(","):
        positive_links.append((s1, mid))

print("Positive links:", len(positive_links))


# ============================================================
# SAMPLE POSITIVE LINKS
# ============================================================

# We don't need all 7.6M for this benchmark.
# 500k is enough to understand blocking behavior.

SAMPLE_SIZE = min(500_000, len(positive_links))

sample = positive_links[:SAMPLE_SIZE]

needed_s1 = set()
needed_s2s3 = set()

for s1, mid in sample:
    needed_s1.add(s1)
    needed_s2s3.add(mid)

print("Sample links:", len(sample))
print("S1 records needed:", len(needed_s1))
print("S2/S3 records needed:", len(needed_s2s3))


# ============================================================
# LOAD REQUIRED S1 RECORDS
# ============================================================

print("\nLoading required S1 records...")

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
            "name": normalize_name(row.business_name),
            "address": normalize_address(row.business_address),
            "country": row.country.lower().strip()
        }

    if len(s1_data) == len(needed_s1):
        break

print("Loaded S1:", len(s1_data))


# ============================================================
# LOAD REQUIRED S2/S3 RECORDS
# ============================================================

print("\nLoading required S2/S3 records...")

other_data = {}

for path in SOURCES:

    print("Scanning:", path)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=200_000,
        keep_default_na=False
    ):

        mask = chunk["entity_id"].isin(needed_s2s3)

        for row in chunk.loc[mask].itertuples(index=False):
            other_data[row.entity_id] = {
                "name": normalize_name(row.business_name),
                "address": normalize_address(row.business_address),
                "country": row.country.lower().strip()
            }

        if len(other_data) == len(needed_s2s3):
            break

print("Loaded S2/S3:", len(other_data))


# ============================================================
# BENCHMARK
# ============================================================

print("\nRunning benchmark...")

name_hits = 0
address_hits = 0
name_country_hits = 0
name_or_address_hits = 0

missing_records = 0

for i, (s1_id, candidate_id) in enumerate(sample):

    a = s1_data.get(s1_id)
    b = other_data.get(candidate_id)

    if a is None or b is None:
        missing_records += 1
        continue

    same_name = (
        a["name"] != ""
        and a["name"] == b["name"]
    )

    same_address = (
        a["address"] != ""
        and a["address"] == b["address"]
    )

    same_name_country = (
        same_name
        and a["country"] == b["country"]
    )

    if same_name:
        name_hits += 1

    if same_address:
        address_hits += 1

    if same_name_country:
        name_country_hits += 1

    if same_name or same_address:
        name_or_address_hits += 1


N = len(sample) - missing_records

print("\n================ RESULTS ================")

print("Usable positive pairs:", N)

print(
    "Exact normalized NAME recall:",
    round(name_hits / N * 100, 2),
    "%"
)

print(
    "Exact normalized ADDRESS recall:",
    round(address_hits / N * 100, 2),
    "%"
)

print(
    "Exact normalized NAME + COUNTRY recall:",
    round(name_country_hits / N * 100, 2),
    "%"
)

print(
    "NAME OR ADDRESS recall:",
    round(name_or_address_hits / N * 100, 2),
    "%"
)

print("Missing records:", missing_records)

print("=========================================")