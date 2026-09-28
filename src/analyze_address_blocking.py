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


print("Loading data...")

s1 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source1.tsv"),
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source2.tsv"),
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_source3.tsv"),
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
    sep="\t",
    dtype=str
)

print("Loaded.")

# ---------------------------------------------------------
# NORMALIZE ADDRESS
# ---------------------------------------------------------

print("Normalizing addresses...")

s1["address_norm"] = s1["business_address"].map(normalize_text)
s2["address_norm"] = s2["business_address"].map(normalize_text)
s3["address_norm"] = s3["business_address"].map(normalize_text)


# ---------------------------------------------------------
# BUILD COUNTRY + ADDRESS INDEX
# ---------------------------------------------------------

print("Building S2 index...")

s2_index = defaultdict(list)

for entity_id, country, address in zip(
    s2["entity_id"],
    s2["country"],
    s2["address_norm"]
):
    if address:
        key = (country, address)
        s2_index[key].append(entity_id)


print("Building S3 index...")

s3_index = defaultdict(list)

for entity_id, country, address in zip(
    s3["entity_id"],
    s3["country"],
    s3["address_norm"]
):
    if address:
        key = (country, address)
        s3_index[key].append(entity_id)


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

print("\nAnalyzing country + exact normalized address...")

candidate_counts = []

total_true_matches = 0
recovered_true_matches = 0

for idx, row in s1.iterrows():

    country = row["country"]
    address = row["address_norm"]

    candidates = set()

    if address:

        key = (country, address)

        candidates.update(
            s2_index.get(key, [])
        )

        candidates.update(
            s3_index.get(key, [])
        )

    candidate_counts.append(len(candidates))

    # Ground truth
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

    recovered_true_matches += len(
        true_matches.intersection(candidates)
    )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

candidate_series = pd.Series(candidate_counts)

print("\n================ RESULTS ================")

print("S1 entities:", len(s1))

print("\nCandidate count statistics:")

print(candidate_series.describe())

print("\nTotal true matches:", total_true_matches)

print(
    "True matches recovered:",
    recovered_true_matches
)

if total_true_matches:
    recall = recovered_true_matches / total_true_matches

    print("Recall:", recall)

print("\nAnalysis complete.")