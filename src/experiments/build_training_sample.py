import os
import pickle
import random
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

ROOT = r"D:\A-Z_ML_contest"

S1_FILE = os.path.join(ROOT, "dataset", "train", "train_source1.tsv")
S2_FILE = os.path.join(ROOT, "dataset", "train", "train_source2.tsv")
S3_FILE = os.path.join(ROOT, "dataset", "train", "train_source3.tsv")
GT_FILE = os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv")

CAND_FILE = os.path.join(
    ROOT, "cache", "train_candidate_pairs.parquet"
)

INDEX_FILE = os.path.join(
    ROOT, "cache", "train_index.pkl"
)

OUT_FILE = os.path.join(
    ROOT, "cache", "training_pairs.parquet"
)

# Keep this reasonably small for the first training run.
N_POSITIVE = 100_000
N_NEGATIVE = 300_000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def split_ids(value):
    """
    Ground truth stores multiple target IDs as comma-separated
    values.
    """
    if pd.isna(value) or str(value).strip() == "":
        return []

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


def group_split(s1_idx):
    """
    Deterministic split by S1 entity.

    0 -> validation
    1-9 -> training
    """
    return zlib.crc32(str(int(s1_idx)).encode()) % 10


# ============================================================
# LOAD S1
# ============================================================

print("=" * 70)
print("BUILD TRAINING SAMPLE")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

print("S1:", len(s1))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

print("GT rows:", len(gt))


# ============================================================
# BUILD POSITIVE PAIRS
# ============================================================

print("\nBuilding positive pairs...")

positive_rows = []

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    targets = split_ids(row.matched_entity_ids)

    for target in targets:
        positive_rows.append(
            (s1_id, target)
        )

print("Total true pairs:", len(positive_rows))

if len(positive_rows) > N_POSITIVE:

    positive_rows = random.sample(
        positive_rows,
        N_POSITIVE
    )

print("Selected positives:", len(positive_rows))


# ============================================================
# MAP S1 ENTITY ID -> INDEX
# ============================================================

print("\nBuilding S1 ID map...")

s1_id_to_idx = dict(
    zip(
        s1["entity_id"],
        range(len(s1))
    )
)

positive_s1_indices = []

for s1_id, target_id in positive_rows:

    idx = s1_id_to_idx.get(s1_id)

    if idx is not None:
        positive_s1_indices.append(
            (idx, target_id)
        )

print(
    "Positive pairs with valid S1:",
    len(positive_s1_indices)
)


# ============================================================
# LOAD CANDIDATE PARQUET
# ============================================================

print("\nOpening candidate parquet...")

pf = pq.ParquetFile(CAND_FILE)

print(
    "Candidate rows:",
    pf.metadata.num_rows
)

print(
    "Row groups:",
    pf.num_row_groups
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

print("\nSampling negative candidates...")

negative_parts = []

needed = N_NEGATIVE

# Randomly choose row groups rather than scanning all 527M rows.
row_groups = list(range(pf.num_row_groups))
random.shuffle(row_groups)

for rg in row_groups:

    if needed <= 0:
        break

    print(
        f"Reading row group {rg} "
        f"(remaining negatives: {needed})"
    )

    table = pf.read_row_group(
        rg,
        columns=[
            "s1_idx",
            "cand_code",
            "n_signals"
        ]
    )

    df = table.to_pandas()

    if len(df) == 0:
        continue

    # Randomly sample from this row group.
    take = min(
        max(needed * 3, 10000),
        len(df)
    )

    sample = df.sample(
        n=take,
        random_state=random.randint(
            0, 2**31 - 1
        )
    )

    negative_parts.append(sample)

    needed -= len(sample)

    print(
        "Collected:",
        N_NEGATIVE - max(needed, 0)
    )


negative = pd.concat(
    negative_parts,
    ignore_index=True
)

if len(negative) > N_NEGATIVE:
    negative = negative.sample(
        n=N_NEGATIVE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

print(
    "\nNegative samples:",
    len(negative)
)


# ============================================================
# REMOVE OBVIOUS TRUE PAIRS FROM NEGATIVES
# ============================================================

print("\nPreparing positive lookup...")

# We cannot cheaply build the complete 7.6M-pair Python set here.
# Instead, the random negative sample will be filtered later
# against a sampled positive lookup where possible.

positive_set = set(
    (s1_id, target_id)
    for s1_id, target_id in positive_rows
)

# Load index used by candidate generation.
print("\nLoading candidate index...")

with open(INDEX_FILE, "rb") as f:
    index_data = pickle.load(f)

rid_ids = index_data["rid_ids"]

print(
    "Candidate record IDs:",
    len(rid_ids)
)


# ============================================================
# CONVERT CAND CODE -> ENTITY ID
# ============================================================

print("\nConverting candidate codes...")

negative["target_entity_id"] = negative[
    "cand_code"
].map(
    lambda x: rid_ids[int(x)]
)

negative["s1_entity_id"] = negative[
    "s1_idx"
].map(
    lambda x: s1.iloc[int(x)]["entity_id"]
)


# Remove sampled pairs that accidentally belong to
# our positive sample.

before = len(negative)

negative = negative[
    ~negative.apply(
        lambda r: (
            r["s1_entity_id"],
            r["target_entity_id"]
        ) in positive_set,
        axis=1
    )
].reset_index(drop=True)

print(
    "Removed sampled positive collisions:",
    before - len(negative)
)


# ============================================================
# CREATE POSITIVE DATAFRAME
# ============================================================

positive = pd.DataFrame(
    positive_s1_indices,
    columns=[
        "s1_idx",
        "target_entity_id"
    ]
)

positive["label"] = 1

# Positive examples do not come from candidate parquet,
# so n_signals is initially unknown.
positive["n_signals"] = 0


# ============================================================
# PREPARE NEGATIVE DATAFRAME
# ============================================================

negative = negative[
    [
        "s1_idx",
        "target_entity_id",
        "n_signals"
    ]
].copy()

negative["label"] = 0


# ============================================================
# COMBINE
# ============================================================

train = pd.concat(
    [
        positive,
        negative
    ],
    ignore_index=True
)

# Add entity IDs for later feature extraction.
train["s1_entity_id"] = train[
    "s1_idx"
].map(
    lambda x: s1.iloc[int(x)]["entity_id"]
)


# ============================================================
# TRAIN / VALIDATION GROUP
# ============================================================

train["split"] = train[
    "s1_idx"
].map(group_split)

train["split"] = np.where(
    train["split"] == 0,
    "valid",
    "train"
)


# ============================================================
# SHUFFLE
# ============================================================

train = train.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

print("\nSaving training sample...")

train.to_parquet(
    OUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("Output:", OUT_FILE)
print("Rows:", len(train))

print(
    "\nLabels:"
)

print(
    train["label"].value_counts()
)

print(
    "\nSplits:"
)

print(
    train["split"].value_counts()
)