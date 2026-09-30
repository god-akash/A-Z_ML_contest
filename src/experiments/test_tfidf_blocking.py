import os
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


TRAIN_DIR = "dataset/train"

SAMPLE_SIZE = 20000


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# LOAD
# =========================================================

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
# NORMALIZE NAMES
# =========================================================

print("\nNormalizing names...")

s1["name_norm"] = s1["business_name"].map(normalize_text)
s2["name_norm"] = s2["business_name"].map(normalize_text)
s3["name_norm"] = s3["business_name"].map(normalize_text)


# =========================================================
# COMBINE S2 + S3
# =========================================================

print("\nCombining S2 and S3...")

sources = pd.concat(
    [
        s2[["entity_id", "name_norm"]],
        s3[["entity_id", "name_norm"]],
    ],
    ignore_index=True
)

# Remove completely empty names
sources = sources[sources["name_norm"] != ""].reset_index(drop=True)

print("Total candidate records:", len(sources))


# =========================================================
# TF-IDF
# =========================================================

print("\nBuilding character TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_features=500000,
    dtype=np.float32
)

source_matrix = vectorizer.fit_transform(
    sources["name_norm"]
)

print("TF-IDF matrix shape:", source_matrix.shape)


# =========================================================
# NEAREST NEIGHBORS
# =========================================================

print("\nBuilding nearest-neighbor index...")

nn = NearestNeighbors(
    n_neighbors=100,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

nn.fit(source_matrix)


# =========================================================
# SAMPLE S1
# =========================================================

sample_indices = np.linspace(
    0,
    len(s1) - 1,
    min(SAMPLE_SIZE, len(s1)),
    dtype=int
)

sample = s1.iloc[sample_indices].copy()

print(
    "\nTesting",
    len(sample),
    "S1 entities..."
)


query_matrix = vectorizer.transform(
    sample["name_norm"]
)

distances, indices = nn.kneighbors(
    query_matrix,
    n_neighbors=100
)


# =========================================================
# EVALUATE
# =========================================================

ks = [10, 20, 50, 100]

hits = {k: 0 for k in ks}

total_true_matches = 0

for local_i, original_index in enumerate(sample_indices):

    gt_value = gt.iloc[original_index]["matched_entity_ids"]

    if pd.isna(gt_value):
        true_matches = set()
    else:
        true_matches = {
            x.strip()
            for x in str(gt_value).split(",")
            if x.strip()
        }

    total_true_matches += len(true_matches)

    retrieved_ids = sources.iloc[
        indices[local_i]
    ]["entity_id"].tolist()

    for k in ks:

        top_k = set(retrieved_ids[:k])

        hits[k] += len(
            true_matches.intersection(top_k)
        )


# =========================================================
# RESULTS
# =========================================================

print("\n================ RESULTS ================")

print("Sample S1:", len(sample))
print("True matches:", total_true_matches)

for k in ks:

    recall = (
        hits[k] / total_true_matches
        if total_true_matches
        else 0
    )

    print(
        f"Top-{k}: "
        f"recovered={hits[k]:,} "
        f"recall={recall:.4%}"
    )

print("\nExperiment complete.")