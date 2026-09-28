import pandas as pd
import re
import unicodedata
from collections import defaultdict
import os
import time

BASE = "dataset"

SOURCES = [
    f"{BASE}/test/test_source2.tsv",
    f"{BASE}/test/test_source3.tsv",
]

S1_FILE = f"{BASE}/test/test_source1.tsv"

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKC", str(x).lower())

    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def tokens(x):
    x = norm(x)

    return [
        t for t in x.split()
        if len(t) >= 3
    ]


def informative_tokens(name, address):
    ts = set(tokens(name) + tokens(address))

    # Remove very common generic business/address words.
    stop = {
        "the", "and", "for",
        "road", "street", "st",
        "rd", "drive", "dr",
        "lane", "ln",
        "avenue", "ave",
        "floor", "city",
        "new", "north", "south",
        "east", "west",
        "limited", "private",
        "pvt", "ltd", "llc",
        "inc", "corp", "company",
        "co", "group",
        "services", "center"
    }

    return [t for t in ts if t not in stop]


# ============================================================
# BUILD INDEX
# ============================================================

print("Building indexes...")

name_index = defaultdict(list)
address_index = defaultdict(list)
token_index = defaultdict(list)

record_count = 0

start = time.time()

for source_file in SOURCES:

    print("\nReading:", source_file)

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            eid = row.entity_id

            n = norm(row.business_name)
            a = norm(row.business_address)

            # Exact indexes
            if n:
                name_index[n].append(eid)

            if a:
                address_index[a].append(eid)

            # Token index
            its = informative_tokens(
                row.business_name,
                row.business_address
            )

            # Only use a few tokens per record.
            # Prefer longer/more distinctive-looking tokens.
            its = sorted(
                its,
                key=lambda x: (-len(x), x)
            )[:4]

            for t in its:
                token_index[t].append(eid)

            record_count += 1

        if record_count % 1_000_000 < 100_000:
            print("Indexed:", f"{record_count:,}")

print("\nTotal indexed:", f"{record_count:,}")
print("Index time:", round(time.time() - start, 1), "sec")


# ============================================================
# MATCH TEST S1
# ============================================================

print("\nMatching S1...")

results = []
candidate_rows = []

s1_count = 0

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=50_000,
    keep_default_na=False
):

    for row in chunk.itertuples(index=False):

        s1 = row.entity_id

        n = norm(row.business_name)
        a = norm(row.business_address)

        candidates = set()

        # ----------------------------------------------------
        # EXACT NAME
        # ----------------------------------------------------

        if n in name_index:
            candidates.update(name_index[n])

        # ----------------------------------------------------
        # EXACT ADDRESS
        # ----------------------------------------------------

        if a in address_index:
            candidates.update(address_index[a])

        # ----------------------------------------------------
        # TOKEN BLOCKING
        # ----------------------------------------------------

        its = sorted(
            informative_tokens(
                row.business_name,
                row.business_address
            ),
            key=lambda x: (-len(x), x)
        )[:4]

        for t in its:
            candidates.update(token_index.get(t, []))

        # Limit pathological candidate explosions
        if len(candidates) > 200:
            candidates = set(list(candidates)[:200])

        # ----------------------------------------------------
        # SAVE CANDIDATES
        # ----------------------------------------------------

        for cid in candidates:
            candidate_rows.append(
                (s1, cid)
            )

        # ----------------------------------------------------
        # SIMPLE MATCHING RULE
        # ----------------------------------------------------

        selected = set()

        # Exact name/address are high confidence.
        exact_name = name_index.get(n, [])
        exact_address = address_index.get(a, [])

        selected.update(exact_name)
        selected.update(exact_address)

        # For token candidates, use a simple shared-token rule.
        # Keep only candidates with at least one informative
        # shared token.
        s1_tokens = set(
            informative_tokens(
                row.business_name,
                row.business_address
            )
        )

        if s1_tokens:

            # We don't have candidate field data here, so token
            # retrieval itself is our weak signal.
            #
            # Keep a small number of token candidates only when
            # there is no exact match.
            if not selected:
                selected.update(
                    list(candidates)[:10]
                )

        for cid in selected:
            results.append(
                (s1, cid)
            )

        s1_count += 1

    print(
        "Processed S1:",
        f"{s1_count:,}",
        "| matches:",
        f"{len(results):,}"
    )


# ============================================================
# DEDUPLICATE
# ============================================================

print("\nDeduplicating...")

results = list(set(results))
candidate_rows = list(set(candidate_rows))


# ============================================================
# OUTPUT
# ============================================================

print("Final matches:", len(results))
print("Final candidates:", len(candidate_rows))

match_df = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "matched_entity_id"
    ]
)

candidate_df = pd.DataFrame(
    candidate_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

match_df.to_csv(
    f"{OUT_DIR}/matching_results.tsv",
    sep="\t",
    index=False
)

candidate_df.to_csv(
    f"{OUT_DIR}/candidate_pairs.tsv",
    sep="\t",
    index=False
)

print("\n================================")
print("SUBMISSION FILES CREATED")
print("================================")

print(
    f"{OUT_DIR}/matching_results.tsv"
)

print(
    f"{OUT_DIR}/candidate_pairs.tsv"
)

print("\nDONE.")