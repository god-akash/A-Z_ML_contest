import os
import re
import unicodedata
import pandas as pd


TRAIN_DIR = "dataset/train"
SAMPLE_SIZE = 20000


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove accents / transliteration to ASCII
    text = (
        unicodedata
        .normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    text = text.lower()

    # punctuation -> spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    # -----------------------------------------------------
    # Remove common legal suffixes
    # -----------------------------------------------------

    suffixes = [
        "private limited",
        "pvt ltd",
        "limited",
        "corporation",
        "corporate",
        "company",
        "gmbh",
        "llp",
        "plc",
        "ltd",
        "llc",
        "inc",
        "corp",
        "co",
    ]

    # longest first
    suffixes = sorted(
        suffixes,
        key=len,
        reverse=True
    )

    changed = True

    while changed:

        changed = False

        for suffix in suffixes:

            if text.endswith(" " + suffix):

                text = text[
                    :-(len(suffix) + 1)
                ].strip()

                changed = True
                break

    return text


print("Loading S1...")

s1 = pd.read_csv(
    os.path.join(
        TRAIN_DIR,
        "train_source1.tsv"
    ),
    sep="\t",
    dtype=str
)

print("Loading S2...")

s2 = pd.read_csv(
    os.path.join(
        TRAIN_DIR,
        "train_source2.tsv"
    ),
    sep="\t",
    dtype=str
)

print("Loading S3...")

s3 = pd.read_csv(
    os.path.join(
        TRAIN_DIR,
        "train_source3.tsv"
    ),
    sep="\t",
    dtype=str
)

print("Loading ground truth...")

gt = pd.read_csv(
    os.path.join(
        TRAIN_DIR,
        "train_ground_truth.tsv"
    ),
    sep="\t",
    dtype=str
).fillna("")


print("Loaded.")


# =========================================================
# NORMALIZE
# =========================================================

print("\nNormalizing names...")

s1["name_norm"] = (
    s1["business_name"]
    .map(normalize)
)

s2["name_norm"] = (
    s2["business_name"]
    .map(normalize)
)

s3["name_norm"] = (
    s3["business_name"]
    .map(normalize)
)


# =========================================================
# BUILD INDEX
# =========================================================

print("\nBuilding S2 index...")

s2_index = {}

for row in s2.itertuples(index=False):

    if not row.name_norm:
        continue

    key = (
        row.country,
        row.name_norm
    )

    if key not in s2_index:
        s2_index[key] = []

    s2_index[key].append(
        row.entity_id
    )


print("Building S3 index...")

s3_index = {}

for row in s3.itertuples(index=False):

    if not row.name_norm:
        continue

    key = (
        row.country,
        row.name_norm
    )

    if key not in s3_index:
        s3_index[key] = []

    s3_index[key].append(
        row.entity_id
    )


# =========================================================
# GROUND TRUTH INDEX
# =========================================================

print("Building ground truth index...")

gt_index = {}

for row in gt.itertuples(index=False):

    if not row.matched_entity_ids:
        gt_index[row.source1_entity_id] = set()

    else:

        gt_index[row.source1_entity_id] = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }


# =========================================================
# SAMPLE
# =========================================================

sample = s1.head(
    min(SAMPLE_SIZE, len(s1))
)


total_true = 0
recovered_true = 0

candidate_counts = []

entities_with_all_matches = 0
entities_with_some_matches = 0


print(
    f"\nTesting {len(sample):,} S1 entities..."
)


# =========================================================
# EVALUATION
# =========================================================

for row in sample.itertuples(index=False):

    key = (
        row.country,
        row.name_norm
    )

    candidates = set()

    candidates.update(
        s2_index.get(key, [])
    )

    candidates.update(
        s3_index.get(key, [])
    )

    candidate_counts.append(
        len(candidates)
    )

    true_matches = gt_index.get(
        row.entity_id,
        set()
    )

    total_true += len(true_matches)

    hits = true_matches.intersection(
        candidates
    )

    recovered_true += len(hits)

    if true_matches:

        if hits:
            entities_with_some_matches += 1

        if hits == true_matches:
            entities_with_all_matches += 1


# =========================================================
# RESULTS
# =========================================================

candidate_series = pd.Series(
    candidate_counts
)

print("\n================ RESULTS ================")

print(
    f"S1 tested: {len(sample):,}"
)

print(
    f"True matches: {total_true:,}"
)

print(
    f"Recovered true matches: {recovered_true:,}"
)

if total_true:

    print(
        "Pair recall:",
        f"{recovered_true / total_true:.4%}"
    )

print(
    "\nS1 entities with at least one true match:"
)

print(
    "Some match recovered:",
    entities_with_some_matches
)

print(
    "ALL matches recovered:",
    entities_with_all_matches
)

print(
    "\nCandidate statistics:"
)

print(
    candidate_series.describe()
)

print(
    "\nCandidate percentiles:"
)

for p in [50, 75, 90, 95, 99, 99.9]:

    print(
        f"{p}%:",
        candidate_series.quantile(p / 100)
    )

print(
    "\nMaximum candidates:",
    candidate_series.max()
)

print("\nAnalysis complete.")