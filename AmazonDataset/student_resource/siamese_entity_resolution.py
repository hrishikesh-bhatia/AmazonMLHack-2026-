import os
import re
import random
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

MAX_VOCAB = 100_000
MAX_LEN = 80

EMBED_DIM = 128
HIDDEN_DIM = 128

BATCH_SIZE = 256
EPOCHS = 3
LR = 1e-3

NEGATIVES_PER_POSITIVE = 2

TEMPERATURE = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("SIAMESE BUSINESS ENTITY RESOLUTION")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)

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


def tokenize(text):

    text = normalize(text)

    return [
        x
        for x in text.split()
        if len(x) >= 2
    ]


def build_text(name, address, country):

    name = "" if pd.isna(name) else str(name)
    address = "" if pd.isna(address) else str(address)
    country = "" if pd.isna(country) else str(country)

    return (
        "name " + name +
        " address " + address +
        " country " + country
    )


# ============================================================
# VOCABULARY
# ============================================================

PAD = "<PAD>"
UNK = "<UNK>"

vocab = {
    PAD: 0,
    UNK: 1
}

token_counts = defaultdict(int)

print("\nBuilding vocabulary...")


def count_tokens(filename):

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            text = build_text(
                row.business_name,
                row.business_address,
                row.country
            )

            for token in tokenize(text):
                token_counts[token] += 1


# Use training sources to construct vocabulary.
count_tokens(TRAIN_S1)
count_tokens(TRAIN_S2)
count_tokens(TRAIN_S3)


most_common = sorted(
    token_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:MAX_VOCAB - 2]


for token, _ in most_common:
    vocab[token] = len(vocab)


print("Vocabulary size:", len(vocab))


# ============================================================
# TEXT -> IDS
# ============================================================

def encode(text):

    ids = [
        vocab.get(token, vocab[UNK])
        for token in tokenize(text)
    ]

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:

        ids += [
            vocab[PAD]
        ] * (
            MAX_LEN - len(ids)
        )

    return ids


# ============================================================
# LOAD TRAINING RECORDS
# ============================================================

print("\nLoading training records...")

records = {}


def load_source(filename):

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            records[row.entity_id] = build_text(
                row.business_name,
                row.business_address,
                row.country
            )


load_source(TRAIN_S1)
load_source(TRAIN_S2)
load_source(TRAIN_S3)

print("Records loaded:", len(records))


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

    matches = row.matched_entity_ids.split(",")

    for match in matches:

        if (
            row.source1_entity_id in records
            and match in records
        ):

            positive_pairs.append(
                (
                    row.source1_entity_id,
                    match
                )
            )


print(
    "Positive pairs:",
    len(positive_pairs)
)


# ============================================================
# NEGATIVE SAMPLING
# ============================================================

all_s2s3 = [
    eid
    for eid in records
    if eid.startswith("S2-")
    or eid.startswith("S3-")
]

print(
    "Candidate negative pool:",
    len(all_s2s3)
)


positive_set = set(positive_pairs)


training_pairs = []


for s1, positive in positive_pairs:

    training_pairs.append(
        (
            s1,
            positive,
            1
        )
    )

    for _ in range(NEGATIVES_PER_POSITIVE):

        negative = random.choice(all_s2s3)

        while (
            negative == positive
            or (s1, negative) in positive_set
        ):

            negative = random.choice(all_s2s3)

        training_pairs.append(
            (
                s1,
                negative,
                0
            )
        )


print(
    "Training pairs:",
    len(training_pairs)
)


# ============================================================
# DATASET
# ============================================================

class EntityDataset(Dataset):

    def __init__(self, pairs):

        self.pairs = pairs

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        s1, s2, label = self.pairs[index]

        x1 = encode(records[s1])
        x2 = encode(records[s2])

        return (
            torch.tensor(
                x1,
                dtype=torch.long
            ),

            torch.tensor(
                x2,
                dtype=torch.long
            ),

            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


dataset = EntityDataset(
    training_pairs
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# SIAMESE ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=vocab[PAD]
        )

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )


    def forward(self, x):

        mask = (
            x != vocab[PAD]
        ).float()

        embedded = self.embedding(x)

        output, _ = self.gru(
            embedded
        )

        mask = mask.unsqueeze(-1)

        pooled = (
            output * mask
        ).sum(dim=1)

        denominator = mask.sum(
            dim=1
        ).clamp(min=1)

        pooled = (
            pooled /
            denominator
        )

        representation = self.projection(
            pooled
        )

        representation = F.normalize(
            representation,
            p=2,
            dim=1
        )

        return representation


# ============================================================
# SIAMESE NETWORK
# ============================================================

class SiameseNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder(
            len(vocab),
            EMBED_DIM,
            HIDDEN_DIM
        )


    def forward(self, x1, x2):

        z1 = self.encoder(x1)

        z2 = self.encoder(x2)

        similarity = F.cosine_similarity(
            z1,
            z2
        )

        return z1, z2, similarity


model = SiameseNetwork().to(DEVICE)


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.5):

        super().__init__()

        self.margin = margin


    def forward(
        self,
        z1,
        z2,
        labels
    ):

        distance = 1 - F.cosine_similarity(
            z1,
            z2
        )

        positive_loss = (
            labels *
            distance.pow(2)
        )

        negative_loss = (
            (1 - labels) *
            F.relu(
                self.margin - distance
            ).pow(2)
        )

        loss = (
            positive_loss +
            negative_loss
        ).mean()

        return loss


criterion = ContrastiveLoss(
    margin=0.5
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    for batch_idx, (
        x1,
        x2,
        labels
    ) in enumerate(loader):

        x1 = x1.to(DEVICE)

        x2 = x2.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        z1, z2, similarity = model(
            x1,
            x2
        )

        loss = criterion(
            z1,
            z2,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch_idx}/{len(loader)} "
                f"Loss {loss.item():.4f}"
            )


    avg_loss = (
        total_loss /
        len(loader)
    )

    print(
        f"\nEpoch {epoch + 1} "
        f"Average Loss: {avg_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model": model.state_dict(),
        "vocab": vocab
    },
    f"{OUTPUT_DIR}/siamese_model.pt"
)

print("\nModel saved.")


# ============================================================
# TEST DATA
# ============================================================

print("\nLoading test S1...")

test_s1 = {}

for chunk in pd.read_csv(
    TEST_S1,
    sep="\t",
    dtype=str,
    chunksize=100_000,
    keep_default_na=False
):

    for row in chunk.itertuples(index=False):

        test_s1[row.entity_id] = build_text(
            row.business_name,
            row.business_address,
            row.country
        )


# ============================================================
# BUILD SIMPLE BLOCKING INDEX
# ============================================================

print("\nBuilding test blocking index...")

token_index = defaultdict(list)


def important_tokens(text):

    ts = set(tokenize(text))

    # Remove generic tokens.
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
        "limited",
        "private",
        "pvt",
        "ltd",
        "llc",
        "inc",
        "corp",
        "company",
        "co",
        "group"
    }

    ts = [
        t
        for t in ts
        if t not in stopwords
    ]

    return sorted(
        ts,
        key=lambda x: -len(x)
    )[:5]


def build_test_index(filename):

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False
    ):

        for row in chunk.itertuples(index=False):

            text = build_text(
                row.business_name,
                row.business_address,
                row.country
            )

            for token in important_tokens(text):

                token_index[token].append(
                    (
                        row.entity_id,
                        text
                    )
                )


build_test_index(TEST_S2)
build_test_index(TEST_S3)


print(
    "Indexed tokens:",
    len(token_index)
)


# ============================================================
# MATCHING
# ============================================================

print("\nGenerating matches...")

matching_results = []

candidate_results = []


model.eval()


def encode_single(text):

    ids = encode(text)

    tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        embedding = model.encoder(
            tensor
        )

    return embedding


# Thresholds.
#
# This is intentionally conservative.
SIMILARITY_THRESHOLD = 0.70

MAX_CANDIDATES = 100


with torch.no_grad():

    for idx, (
        s1_id,
        s1_text
    ) in enumerate(test_s1.items()):

        candidates = {}

        for token in important_tokens(
            s1_text
        ):

            for eid, text in token_index.get(
                token,
                []
            ):

                if eid not in candidates:

                    candidates[eid] = text

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


        # ----------------------------------------------------
        # Candidate output
        # ----------------------------------------------------

        candidate_results.append(
            (
                s1_id,
                list(candidates.keys())
            )
        )


        if not candidates:

            matching_results.append(
                (
                    s1_id,
                    []
                )
            )

            continue


        # ----------------------------------------------------
        # Encode S1
        # ----------------------------------------------------

        s1_embedding = encode_single(
            s1_text
        )


        candidate_ids = list(
            candidates.keys()
        )

        candidate_texts = [
            candidates[x]
            for x in candidate_ids
        ]


        # ----------------------------------------------------
        # Encode candidates
        # ----------------------------------------------------

        candidate_tensor = torch.tensor(
            [
                encode(text)
                for text in candidate_texts
            ],
            dtype=torch.long,
            device=DEVICE
        )


        embeddings = model.encoder(
            candidate_tensor
        )


        similarities = F.cosine_similarity(
            s1_embedding,
            embeddings
        )


        selected = []


        for eid, score in zip(
            candidate_ids,
            similarities.cpu().tolist()
        ):

            if score >= SIMILARITY_THRESHOLD:

                selected.append(
                    eid
                )


        matching_results.append(
            (
                s1_id,
                selected
            )
        )


        if idx % 10_000 == 0:

            print(
                f"Processed "
                f"{idx:,}/"
                f"{len(test_s1):,}"
            )


# ============================================================
# WRITE MATCHING RESULTS
# ============================================================

print("\nWriting matching_results.tsv...")

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

    for s1_id, matches in matching_results:

        f.write(
            s1_id +
            "\t" +
            ",".join(matches) +
            "\n"
        )


# ============================================================
# WRITE CANDIDATE PAIRS
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

    for s1_id, candidates in candidate_results:

        f.write(
            s1_id +
            "\t" +
            ",".join(candidates) +
            "\n"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "Matching:",
    f"{OUTPUT_DIR}/matching_results.tsv"
)

print(
    "Candidates:",
    f"{OUTPUT_DIR}/candidate_pairs.tsv"
)

print(
    "Model:",
    f"{OUTPUT_DIR}/siamese_model.pt"
)