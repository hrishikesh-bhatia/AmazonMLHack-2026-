# preprocess.py

import os
import re
import unicodedata
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATASET = "dataset"
OUTPUT = "preprocessed"

os.makedirs(OUTPUT, exist_ok=True)

CHUNK_SIZE = 200_000


FILES = {
    "train_source1": f"{DATASET}/train/train_source1.tsv",
    "train_source2": f"{DATASET}/train/train_source2.tsv",
    "train_source3": f"{DATASET}/train/train_source3.tsv",

    "test_source1": f"{DATASET}/test/test_source1.tsv",
    "test_source2": f"{DATASET}/test/test_source2.tsv",
    "test_source3": f"{DATASET}/test/test_source3.tsv",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize(
        "NFKC",
        value
    )

    # Lowercase
    value = value.lower()

    # Replace punctuation with spaces.
    #
    # \w keeps Unicode characters, numbers and underscores.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Normalize whitespace
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# TOKENIZATION
# ============================================================

STOPWORDS = {
    "the",
    "and",
    "for",
    "of",

    # Address terms
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
    "boulevard",
    "blvd",
    "highway",
    "hwy",

    # Business suffixes
    "limited",
    "private",
    "pvt",
    "ltd",
    "llc",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",

    # Generic business terms
    "group",
    "services",
    "service",
    "center",
    "centre",

    # Generic location words
    "city",
    "north",
    "south",
    "east",
    "west"
}


def make_tokens(text):

    if not text:
        return ""

    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if len(token) >= 2
        and token not in STOPWORDS
    ]

    return " ".join(tokens)


# ============================================================
# NUMERIC TOKEN EXTRACTION
# ============================================================

def extract_numbers(text):

    if not text:
        return ""

    tokens = text.split()

    numbers = []

    for token in tokens:

        if any(
            character.isdigit()
            for character in token
        ):
            numbers.append(token)

    return " ".join(numbers)


# ============================================================
# RECORD PREPROCESSING
# ============================================================

def preprocess_chunk(df):

    # Make sure missing values are strings.
    for column in [
        "business_name",
        "business_address",
        "country"
    ]:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
            )


    # --------------------------------------------------------
    # Normalized fields
    # --------------------------------------------------------

    df["name_normalized"] = (
        df["business_name"]
        .map(normalize_text)
    )

    df["address_normalized"] = (
        df["business_address"]
        .map(normalize_text)
    )

    df["country_normalized"] = (
        df["country"]
        .map(normalize_text)
    )


    # --------------------------------------------------------
    # Token fields
    # --------------------------------------------------------

    df["name_tokens"] = (
        df["name_normalized"]
        .map(make_tokens)
    )

    df["address_tokens"] = (
        df["address_normalized"]
        .map(make_tokens)
    )


    # --------------------------------------------------------
    # Numeric address tokens
    # --------------------------------------------------------

    df["address_numbers"] = (
        df["address_normalized"]
        .map(extract_numbers)
    )


    # --------------------------------------------------------
    # Useful lengths
    # --------------------------------------------------------

    df["name_length"] = (
        df["name_normalized"]
        .str.len()
        .astype("int32")
    )

    df["address_length"] = (
        df["address_normalized"]
        .str.len()
        .astype("int32")
    )

    df["name_token_count"] = (
        df["name_tokens"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype("int16")
    )

    df["address_token_count"] = (
        df["address_tokens"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype("int16")
    )


    return df


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    name,
    input_path
):

    print("\n" + "=" * 70)

    print(
        "Processing:",
        input_path
    )

    print("=" * 70)


    output_path = os.path.join(
        OUTPUT,
        name + ".parquet"
    )


    # Remove old output if it exists.
    if os.path.exists(output_path):

        os.remove(output_path)


    first_chunk = True

    total_rows = 0


    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False
        )
    ):

        print(
            f"Chunk {chunk_number + 1}"
        )

        chunk = preprocess_chunk(
            chunk
        )


        # ----------------------------------------------------
        # Write parquet
        # ----------------------------------------------------

        #
        # pandas does not append to parquet consistently
        # across engines, so we collect temporary parquet
        # chunks and combine them at the end.
        #

        chunk_path = os.path.join(
            OUTPUT,
            f".{name}_chunk_{chunk_number}.parquet"
        )

        chunk.to_parquet(
            chunk_path,
            index=False
        )


        total_rows += len(chunk)


        print(
            "Rows processed:",
            f"{total_rows:,}"
        )


    # --------------------------------------------------------
    # Combine chunks
    # --------------------------------------------------------

    print(
        "\nCombining processed chunks..."
    )


    chunk_files = sorted(
        [
            os.path.join(
                OUTPUT,
                x
            )
            for x in os.listdir(OUTPUT)
            if x.startswith(
                f".{name}_chunk_"
            )
            and x.endswith(".parquet")
        ]
    )


    if not chunk_files:

        print(
            "ERROR: No chunks generated."
        )

        return


    tables = []

    for chunk_file in chunk_files:

        tables.append(
            pd.read_parquet(
                chunk_file
            )
        )


    final_df = pd.concat(
        tables,
        ignore_index=True
    )


    final_df.to_parquet(
        output_path,
        index=False
    )


    # --------------------------------------------------------
    # Remove temporary chunks
    # --------------------------------------------------------

    for chunk_file in chunk_files:

        os.remove(
            chunk_file
        )


    print(
        "\nSaved:",
        output_path
    )

    print(
        "Rows:",
        f"{len(final_df):,}"
    )

    print(
        "Columns:",
        list(final_df.columns)
    )


    del final_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nAmazon ML Challenge"
    )

    print(
        "Business Entity Resolution"
    )

    print(
        "Preprocessing pipeline"
    )


    for name, path in FILES.items():

        if not os.path.exists(path):

            print(
                "\nWARNING: File not found:",
                path
            )

            continue


        process_file(
            name,
            path
        )


    print("\n" + "=" * 70)

    print(
        "PREPROCESSING COMPLETE"
    )

    print("=" * 70)

    print(
        "\nProcessed files are in:"
    )

    print(
        os.path.abspath(OUTPUT)
    )