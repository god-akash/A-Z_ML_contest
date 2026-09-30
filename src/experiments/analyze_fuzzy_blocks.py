import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd


TRAIN_DIR = "dataset/train"


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_keys(name):
    """
    Generate cheap blocking keys from a normalized business name.
    """

    if not name:
        return []

    tokens = name.split()

    keys = set()

    # Full first token prefix
    if tokens:
        first = tokens[0]

        if len(first) >= 3:
            keys.add(first[:3])

        if len(first) >= 4:
            keys.add(first[:4])

        if len(first) >= 5:
            keys.add(first[:5])

    # Individual token prefixes
    for token in tokens:

        if len(token) >= 4:
            keys.add(token[:4])

        if len(token) >= 5:
            keys.add(token[:5])

    # First two tokens
    if len(tokens) >= 2:

        combined = tokens[0] + tokens[1]

        if len(combined) >= 5:
            keys.add(combined[:5])

    return list(keys)


print("Loading S1...")

s1 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source1.tsv"),
    sep="\t",
    dtype=str
)

print("Loading S2...")

s2 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source2.tsv"),
    sep="\t",
    dtype=str
)

print("Loading S3...")

s3 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source3.tsv"),
    sep="\t",
    dtype=str
)

print("Loading ground truth...")

gt = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
    sep="\t",
    dtype=str
)

print("Loaded.")


# =========================================================
# NORMALIZE
# =========================================================

print("\nNormalizing names...")

s1["name_norm"] = s1["business_name"].map(normalize_text)
s2["name_norm"] = s2["business_name"].map(normalize_text)
s3["name_norm"] = s3["business_name"].map(normalize_text)


# =========================================================
# BUILD BLOCK INDEX
# =========================================================

print("\nBuilding S2 block index...")

s2_blocks = defaultdict(list)

for entity_id, country, name in zip(
    s2["entity_id"],
    s2["country"],
    s2["name_norm"]
):

    for key in make_keys(name):

        # Country-aware key
        block_key = (country, key)

        s2_blocks[block_key].append(entity_id)


print("Building S3 block index...")

s3_blocks = defaultdict(list)

for entity_id, country, name in zip(
    s3["entity_id"],
    s3["country"],
    s3["name_norm"]
):

    for key in make_keys(name):

        block_key = (country, key)

        s3_blocks[block_key].append(entity_id)


# =========================================================
# BLOCK STATISTICS
# =========================================================

print("\nCalculating block statistics...")

s2_sizes = pd.Series(
    [len(v) for v in s2_blocks.values()]
)

s3_sizes = pd.Series(
    [len(v) for v in s3_blocks.values()]
)

print("\nS2 block statistics:")
print(s2_sizes.describe())

print("\nS3 block statistics:")
print(s3_sizes.describe())


# =========================================================
# SAMPLE S1
# =========================================================

SAMPLE_SIZE = 20000

sample_indices = range(
    min(SAMPLE_SIZE, len(s1))
)

print(
    f"\nTesting first {len(list(sample_indices))} S1 entities..."
)


# =========================================================
# EVALUATE BLOCK RECALL
# =========================================================

total_true_matches = 0

recovered = {
    10: 0,
    25: 0,
    50: 0,
    100: 0,
    250: 0,
    500: 0,
}

candidate_counts = []


for idx in sample_indices:

    row = s1.iloc[idx]

    country = row["country"]
    name = row["name_norm"]

    candidates = set()

    for key in make_keys(name):

        block_key = (country, key)

        candidates.update(
            s2_blocks.get(block_key, [])
        )

        candidates.update(
            s3_blocks.get(block_key, [])
        )

    candidate_counts.append(len(candidates))

    # -----------------------------------------
    # Ground truth
    # -----------------------------------------

    gt_value = gt.iloc[idx]["matched_entity_ids"]

    if pd.isna(gt_value):

        true_matches = set()

    else:

        true_matches = {
            x.strip()
            for x in str(gt_value).split(",")
            if x.strip()
        }

    total_true_matches += len(true_matches)

    hit_count = len(
        true_matches.intersection(candidates)
    )

    # Since this is only blocking,
    # every candidate is retained.
    #
    # Record recall at various candidate
    # caps.

    for k in recovered:

        # We cannot rank yet, so only count
        # whether the full block contains matches.
        if hit_count > 0:
            recovered[k] += hit_count


# =========================================================
# RESULTS
# =========================================================

candidate_series = pd.Series(candidate_counts)

print("\n================ RESULTS ================")

print(
    "Average candidates:",
    candidate_series.mean()
)

print(
    "Median candidates:",
    candidate_series.median()
)

print(
    "Maximum candidates:",
    candidate_series.max()
)

print("\nCandidate distribution:")

print(
    candidate_series.describe()
)

print(
    "\nTotal true matches in sample:",
    total_true_matches
)

print(
    "True matches recovered by blocks:",
    recovered[10]
)

if total_true_matches:

    print(
        "Blocking recall:",
        recovered[10] / total_true_matches
    )

print("\nAnalysis complete.")
