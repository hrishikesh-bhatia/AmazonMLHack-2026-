import os
import re
import random
import unicodedata
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

BASE = "dataset"

TRAIN_S1 = f"{BASE}/train/train_source1.tsv"
TRAIN_S2 = f"{BASE}/train/train_source2.tsv"
TRAIN_S3 = f"{BASE}/train/train_source3.tsv"

GROUND_TRUTH = f"{BASE}/train/train_ground_truth.tsv"

TEST_S1 = f"{BASE}/test/test_source1.tsv"
TEST_S2 = f"{BASE}/test/test_source2.tsv"
TEST_S3 = f"{BASE}/test/test_source3.tsv"

OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

# Number of positive pairs used for training.
# Increase if you have time/RAM.
MAX_POSITIVE_PAIRS = 300_000

# Number of negative examples per positive.
NEGATIVES_PER_POSITIVE = 2

# Maximum candidates considered per S1.
MAX_CANDIDATES = 100

# Final probability threshold.
MATCH_THRESHOLD = 0.50


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def get_tokens(text):

    text = normalize(text)

    return {
        token
        for token in text.split()
        if len(token) >= 2
    }


def get_numeric_tokens(text):

    return {
        token
        for token in normalize(text).split()
        if any(c.isdigit() for c in token)
    }


def build_record(
    entity_id,
    name,
    address,
    country
):

    return {
        "entity_id": entity_id,
        "name": "" if pd.isna(name) else str(name),
        "address": "" if pd.isna(address) else str(address),
        "country": "" if pd.isna(country) else str(country)
    }


# ============================================================
# LOAD RECORDS
# ============================================================

print("=" * 70)
print("XGBOOST BUSINESS ENTITY RESOLUTION")
print("=" * 70)


# We only load the training data required to construct
# positive/negative training pairs.

print("\nLoading training data...")


s1_train = {}
s23_train = {}


def load_file(
    filename,
    destination
):

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            destination[row.entity_id] = build_record(
                row.entity_id,
                row.business_name,
                row.business_address,
                row.country
            )


load_file(
    TRAIN_S1,
    s1_train
)

print(
    "S1 records:",
    len(s1_train)
)


load_file(
    TRAIN_S2,
    s23_train
)

print(
    "S2 records:",
    len(s23_train)
)


load_file(
    TRAIN_S3,
    s23_train
)

print(
    "S2 + S3 records:",
    len(s23_train)
)


# ============================================================
# GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")


gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


positive_pairs = []


for row in gt.itertuples(index=False):

    if not row.matched_entity_ids:
        continue

    for match_id in row.matched_entity_ids.split(","):

        if (
            row.source1_entity_id in s1_train
            and match_id in s23_train
        ):

            positive_pairs.append(
                (
                    row.source1_entity_id,
                    match_id
                )
            )


random.seed(SEED)

random.shuffle(
    positive_pairs
)

positive_pairs = positive_pairs[
    :MAX_POSITIVE_PAIRS
]


print(
    "Positive pairs:",
    len(positive_pairs)
)


positive_set = set(
    positive_pairs
)


# ============================================================
# CANDIDATE BLOCKING INDEX
# ============================================================

print("\nBuilding blocking index...")


token_index = defaultdict(list)


def important_tokens(record):

    name_tokens = get_tokens(
        record["name"]
    )

    address_tokens = get_tokens(
        record["address"]
    )

    tokens = (
        name_tokens |
        address_tokens
    )

    # Generic/common tokens.
    stopwords = {
        "the",
        "and",
        "for",
        "road",
        "street",
        "st",
        "rd",
        "drive",
        "dr",
        "lane",
        "ln",
        "avenue",
        "ave",
        "floor",
        "city",
        "new",
        "north",
        "south",
        "east",
        "west",
        "limited",
        "private",
        "pvt",
        "ltd",
        "llc",
        "inc",
        "corp",
        "company",
        "co",
        "group",
        "services",
        "center"
    }

    tokens = [
        x
        for x in tokens
        if x not in stopwords
    ]

    # Prefer longer tokens because they are usually
    # more informative.
    tokens.sort(
        key=lambda x: -len(x)
    )

    return tokens[:5]


for entity_id, record in s23_train.items():

    for token in important_tokens(record):

        token_index[token].append(
            entity_id
        )


print(
    "Indexed tokens:",
    len(token_index)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def safe_jaccard(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def safe_overlap(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b)
    )


def create_features(
    r1,
    r2
):

    name1 = normalize(
        r1["name"]
    )

    name2 = normalize(
        r2["name"]
    )

    addr1 = normalize(
        r1["address"]
    )

    addr2 = normalize(
        r2["address"]
    )

    country1 = normalize(
        r1["country"]
    )

    country2 = normalize(
        r2["country"]
    )


    # ----------------------------
    # TOKENS
    # ----------------------------

    name_tokens_1 = get_tokens(
        name1
    )

    name_tokens_2 = get_tokens(
        name2
    )

    addr_tokens_1 = get_tokens(
        addr1
    )

    addr_tokens_2 = get_tokens(
        addr2
    )


    # ----------------------------
    # NUMBERS
    # ----------------------------

    num1 = get_numeric_tokens(
        addr1
    )

    num2 = get_numeric_tokens(
        addr2
    )


    # ----------------------------
    # NAME FEATURES
    # ----------------------------

    name_jaccard = safe_jaccard(
        name_tokens_1,
        name_tokens_2
    )

    name_overlap = safe_overlap(
        name_tokens_1,
        name_tokens_2
    )

    name_ratio = ratio(
        name1,
        name2
    ) / 100.0

    name_token_sort = token_sort_ratio(
        name1,
        name2
    ) / 100.0

    name_token_set = token_set_ratio(
        name1,
        name2
    ) / 100.0


    # ----------------------------
    # ADDRESS FEATURES
    # ----------------------------

    address_jaccard = safe_jaccard(
        addr_tokens_1,
        addr_tokens_2
    )

    address_overlap = safe_overlap(
        addr_tokens_1,
        addr_tokens_2
    )

    address_ratio = ratio(
        addr1,
        addr2
    ) / 100.0

    address_token_sort = token_sort_ratio(
        addr1,
        addr2
    ) / 100.0

    address_token_set = token_set_ratio(
        addr1,
        addr2
    ) / 100.0


    # ----------------------------
    # COUNTRY
    # ----------------------------

    country_match = (
        int(
            country1 != ""
            and country1 == country2
        )
    )


    # ----------------------------
    # NUMERIC ADDRESS
    # ----------------------------

    numeric_jaccard = safe_jaccard(
        num1,
        num2
    )

    numeric_overlap = safe_overlap(
        num1,
        num2
    )


    # ----------------------------
    # EXACT MATCHES
    # ----------------------------

    exact_name = int(
        name1 != ""
        and name1 == name2
    )

    exact_address = int(
        addr1 != ""
        and addr1 == addr2
    )


    # ----------------------------
    # LENGTH FEATURES
    # ----------------------------

    name_length_diff = abs(
        len(name1) -
        len(name2)
    )

    address_length_diff = abs(
        len(addr1) -
        len(addr2)
    )


    # ----------------------------
    # TOKEN COUNTS
    # ----------------------------

    shared_name_tokens = len(
        name_tokens_1 &
        name_tokens_2
    )

    shared_address_tokens = len(
        addr_tokens_1 &
        addr_tokens_2
    )


    return [
        name_jaccard,
        name_overlap,
        name_ratio,
        name_token_sort,
        name_token_set,

        address_jaccard,
        address_overlap,
        address_ratio,
        address_token_sort,
        address_token_set,

        country_match,

        numeric_jaccard,
        numeric_overlap,

        exact_name,
        exact_address,

        name_length_diff,
        address_length_diff,

        len(name_tokens_1),
        len(name_tokens_2),

        len(addr_tokens_1),
        len(addr_tokens_2),

        shared_name_tokens,
        shared_address_tokens
    ]


FEATURE_NAMES = [
    "name_jaccard",
    "name_overlap",
    "name_ratio",
    "name_token_sort",
    "name_token_set",

    "address_jaccard",
    "address_overlap",
    "address_ratio",
    "address_token_sort",
    "address_token_set",

    "country_match",

    "numeric_jaccard",
    "numeric_overlap",

    "exact_name",
    "exact_address",

    "name_length_diff",
    "address_length_diff",

    "name_token_count_1",
    "name_token_count_2",

    "address_token_count_1",
    "address_token_count_2",

    "shared_name_tokens",
    "shared_address_tokens"
]


# ============================================================
# TRAINING NEGATIVES
# ============================================================

print("\nGenerating negative examples...")


all_entities = list(
    s23_train.keys()
)


training_X = []
training_y = []


for index, (
    s1_id,
    positive_id
) in enumerate(
    positive_pairs
):

    r1 = s1_train[s1_id]

    # Positive example
    r2 = s23_train[positive_id]

    training_X.append(
        create_features(
            r1,
            r2
        )
    )

    training_y.append(1)


    # Negative examples
    negatives_added = 0

    while (
        negatives_added <
        NEGATIVES_PER_POSITIVE
    ):

        negative_id = random.choice(
            all_entities
        )

        if (
            negative_id == positive_id
            or
            (
                s1_id,
                negative_id
            ) in positive_set
        ):
            continue


        r2 = s23_train[
            negative_id
        ]


        training_X.append(
            create_features(
                r1,
                r2
            )
        )

        training_y.append(0)

        negatives_added += 1


    if index % 10_000 == 0:

        print(
            f"Processed "
            f"{index:,}/"
            f"{len(positive_pairs):,}"
        )


X = np.asarray(
    training_X,
    dtype=np.float32
)

y = np.asarray(
    training_y,
    dtype=np.int32
)


print(
    "\nTraining matrix:",
    X.shape
)

print(
    "Positive:",
    int(y.sum())
)

print(
    "Negative:",
    int((y == 0).sum())
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


# ============================================================
# XGBOOST
# ============================================================

print("\nTraining XGBoost...")


model = XGBClassifier(

    n_estimators=500,

    max_depth=7,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    random_state=SEED,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_valid, y_valid)
    ],
    verbose=True
)


# ============================================================
# VALIDATION
# ============================================================

print("\nValidation...")


valid_prob = model.predict_proba(
    X_valid
)[:, 1]


# F0.5 is more precision-focused.
best_threshold = 0.5
best_f05 = 0


for threshold in np.arange(
    0.10,
    0.91,
    0.02
):

    pred = (
        valid_prob >= threshold
    ).astype(int)


    precision, recall, fbeta, _ = (
        precision_recall_fscore_support(
            y_valid,
            pred,
            beta=0.5,
            average="binary",
            zero_division=0
        )
    )


    if fbeta > best_f05:

        best_f05 = fbeta
        best_threshold = threshold


print(
    "\nBest threshold:",
    round(best_threshold, 3)
)

print(
    "Validation F0.5:",
    round(best_f05, 5)
)


pred = (
    valid_prob >= best_threshold
).astype(int)


print(
    classification_report(
        y_valid,
        pred,
        digits=4
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    f"{OUTPUT_DIR}/xgboost_entity_resolution.json"
)


# ============================================================
# RELEASE TRAINING DATA
# ============================================================

del X
del y
del X_train
del X_valid
del y_train
del y_valid
del training_X
del training_y


# ============================================================
# TEST INDEX
# ============================================================

print("\nLoading test S2/S3...")


test_index = defaultdict(list)


def add_test_source(filename):

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            record = build_record(
                row.entity_id,
                row.business_name,
                row.business_address,
                row.country
            )

            test_index[row.entity_id] = record


add_test_source(
    TEST_S2
)

print(
    "Test S2 loaded:",
    len(test_index)
)


add_test_source(
    TEST_S3
)

print(
    "Test S2+S3 loaded:",
    len(test_index)
)


# ============================================================
# BUILD TEST TOKEN INDEX
# ============================================================

print("\nBuilding test token index...")


test_token_index = defaultdict(list)


for entity_id, record in test_index.items():

    for token in important_tokens(record):

        test_token_index[token].append(
            entity_id
        )


print(
    "Test blocking tokens:",
    len(test_token_index)
)


# ============================================================
# LOAD TEST S1
# ============================================================

print("\nMatching test S1...")


test_s1_count = 0

matching_output = []

candidate_output = []


for chunk in pd.read_csv(
    TEST_S1,
    sep="\t",
    dtype=str,
    chunksize=50_000,
    keep_default_na=False
):

    for row in chunk.itertuples(index=False):

        s1 = build_record(
            row.entity_id,
            row.business_name,
            row.business_address,
            row.country
        )


        # ----------------------------------------------------
        # CANDIDATE GENERATION
        # ----------------------------------------------------

        candidates = set()


        for token in important_tokens(s1):

            ids = test_token_index.get(
                token,
                []
            )


            # Avoid huge candidate explosions.
            if len(ids) > 5000:

                continue


            for entity_id in ids:

                candidates.add(
                    entity_id
                )


                if (
                    len(candidates)
                    >= MAX_CANDIDATES
                ):

                    break


            if (
                len(candidates)
                >= MAX_CANDIDATES
            ):

                break


        candidates = list(
            candidates
        )[:MAX_CANDIDATES]


        candidate_output.append(
            (
                s1["entity_id"],
                candidates
            )
        )


        if not candidates:

            matching_output.append(
                (
                    s1["entity_id"],
                    []
                )
            )

            test_s1_count += 1

            continue


        # ----------------------------------------------------
        # FEATURE GENERATION
        # ----------------------------------------------------

        feature_rows = []

        valid_candidates = []


        for candidate_id in candidates:

            candidate = test_index[
                candidate_id
            ]

            features = create_features(
                s1,
                candidate
            )

            feature_rows.append(
                features
            )

            valid_candidates.append(
                candidate_id
            )


        feature_matrix = np.asarray(
            feature_rows,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # XGBOOST PREDICTION
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                feature_matrix
            )[:, 1]
        )


        matches = []


        for candidate_id, probability in zip(
            valid_candidates,
            probabilities
        ):

            if probability >= best_threshold:

                matches.append(
                    candidate_id
                )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        matching_output.append(
            (
                s1["entity_id"],
                matches
            )
        )


        test_s1_count += 1


    print(
        "Processed test S1:",
        f"{test_s1_count:,}"
    )


# ============================================================
# WRITE MATCHING RESULTS
# ============================================================

print(
    "\nWriting matching_results.tsv..."
)


with open(
    f"{OUTPUT_DIR}/matching_results.tsv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    f.write(
        "source1_entity_id\t"
        "matched_entity_ids\n"
    )


    for s1_id, matches in matching_output:

        f.write(
            s1_id +
            "\t" +
            ",".join(matches) +
            "\n"
        )


# ============================================================
# WRITE CANDIDATE RESULTS
# ============================================================

print(
    "Writing candidate_pairs.tsv..."
)


with open(
    f"{OUTPUT_DIR}/candidate_pairs.tsv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    f.write(
        "source1_entity_id\t"
        "candidate_entity_ids\n"
    )


    for s1_id, candidates in candidate_output:

        f.write(
            s1_id +
            "\t" +
            ",".join(candidates) +
            "\n"
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\nFeature importance:"
)


importance = model.feature_importances_


for name, value in sorted(
    zip(
        FEATURE_NAMES,
        importance
    ),
    key=lambda x: -x[1]
):

    print(
        f"{name:30s} "
        f"{value:.4f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)

print("DONE")

print("=" * 70)

print(
    "Matching file:",
    f"{OUTPUT_DIR}/matching_results.tsv"
)

print(
    "Candidate file:",
    f"{OUTPUT_DIR}/candidate_pairs.tsv"
)

print(
    "Model:",
    f"{OUTPUT_DIR}/xgboost_entity_resolution.json"
)

print(
    "Validation F0.5:",
    round(best_f05, 5)
)

print(
    "Threshold:",
    round(best_threshold, 3)
)

print("=" * 70)