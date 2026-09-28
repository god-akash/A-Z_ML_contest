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

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace & with and
    text = text.replace("&", " and ")

    # Keep letters/numbers, turn punctuation into spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("Loading S1...")

s1 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source1.tsv"),
    sep="\t",
    dtype=str,
)

print("Loading S2...")

s2 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source2.tsv"),
    sep="\t",
    dtype=str,
)

print("Loading S3...")

s3 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source3.tsv"),
    sep="\t",
    dtype=str,
)

print("Loading ground truth...")

gt = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
    sep="\t",
    dtype=str,
)

print("\nLoaded.")

# ---------------------------------------------------------
# NORMALIZE BUSINESS NAMES
# ---------------------------------------------------------

print("\nNormalizing names...")

s1["name_norm"] = s1["business_name"].map(normalize_text)
s2["name_norm"] = s2["business_name"].map(normalize_text)
s3["name_norm"] = s3["business_name"].map(normalize_text)

# ---------------------------------------------------------
# BUILD NAME INDEX
# ---------------------------------------------------------

print("Building S2 name index...")

s2_index = defaultdict(list)

for entity_id, name in zip(s2["entity_id"], s2["name_norm"]):
    if name:
        s2_index[name].append(entity_id)

print("Building S3 name index...")

s3_index = defaultdict(list)

for entity_id, name in zip(s3["entity_id"], s3["name_norm"]):
    if name:
        s3_index[name].append(entity_id)


# ---------------------------------------------------------
# ANALYZE CANDIDATES
# ---------------------------------------------------------

print("\nAnalyzing exact normalized-name candidates...")

candidate_counts = []

covered = 0
total = 0

for idx, row in s1.iterrows():

    s1_id = row["entity_id"]
    name = row["name_norm"]

    candidates = set()

    if name:
        candidates.update(s2_index.get(name, []))
        candidates.update(s3_index.get(name, []))

    candidate_counts.append(len(candidates))

    # Ground truth
    gt_row = gt.iloc[idx]["matched_entity_ids"]

    if pd.isna(gt_row):
        true_matches = set()
    else:
        true_matches = {
            x.strip()
            for x in str(gt_row).split(",")
            if x.strip()
        }

    total += len(true_matches)

    if true_matches.intersection(candidates):
        covered += len(true_matches.intersection(candidates))


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

candidate_series = pd.Series(candidate_counts)

print("\n================ RESULTS ================")

print("S1 entities:", len(s1))

print("\nCandidate count statistics:")
print(candidate_series.describe())

print("\nCandidate count distribution:")
print(candidate_series.value_counts().sort_index().head(30))

print("\nTotal true matches:", total)
print("True matches recovered:", covered)

if total > 0:
    print("Recall:", covered / total)

print("\nAnalysis complete.")