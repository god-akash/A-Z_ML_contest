# src/build_training_pairs.py
import os
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zlib

ROOT = r"D:\A-Z_ML_contest"
GT_FILE = os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv")
CAND_FILE = os.path.join(ROOT, "cache", "train_candidate_pairs.parquet")
ID_MAP_FILE = os.path.join(ROOT, "cache", "train_id_maps.pkl")
OUT_FILE = os.path.join(ROOT, "cache", "training_pairs.parquet")

N_NEGATIVE_PER_POSITIVE = 4     # negatives sampled per positive kept, not a fixed global count
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def group_split(s1_idx: int) -> str:
    """Deterministic split BY S1 ENTITY. Every candidate row for the same S1
    entity gets the same split, since they all share this s1_idx -- this is
    what actually prevents entity-level leakage (row-level splits don't)."""
    return "valid" if zlib.crc32(str(int(s1_idx)).encode()) % 10 == 0 else "train"


print("=" * 70)
print("BUILD TRAINING PAIRS (labeled directly from candidate parquet)")
print("=" * 70)

print("\nLoading id maps...")
with open(ID_MAP_FILE, "rb") as f:
    id_maps = pickle.load(f)
s1_ids = id_maps["s1_ids"]         # position -> S1 entity_id string
rid_ids = id_maps["rid_ids"]       # position -> S2/S3 entity_id string
s1_id_to_idx = {eid: i for i, eid in enumerate(s1_ids)}
rid_to_code = {eid: c for c, eid in enumerate(rid_ids)}
print(f"S1: {len(s1_ids):,}  S2/S3 referenced: {len(rid_ids):,}")

print("\nLoading FULL ground truth (not a sample -- this must be complete "
      "or negatives can be silently mislabeled)...")
gt = pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["matched_entity_ids"].str.strip() != ""]

# Build the full true-pair set as (s1_idx, cand_code) integer pairs, encoded
# into a single int64 key for fast vectorized membership checks against the
# candidate parquet (avoids slow row-wise .apply() with Python tuples).
CODE_MULT = len(rid_ids) + 1
true_keys = set()
n_true_total = 0
for r in gt.itertuples(index=False):
    s1i = s1_id_to_idx.get(r.source1_entity_id)
    if s1i is None:
        continue
    for other in r.matched_entity_ids.split(","):
        other = other.strip()
        n_true_total += 1
        c = rid_to_code.get(other)
        if c is not None:
            true_keys.add(s1i * CODE_MULT + c)
print(f"True pairs total: {n_true_total:,}  "
      f"reachable via candidate-parquet id maps: {len(true_keys):,}")

print("\nStreaming candidate parquet, labeling each row against full ground truth...")
pf = pq.ParquetFile(CAND_FILE)
pos_parts, neg_parts = [], []
n_pos_seen, n_neg_seen = 0, 0

for rg in range(pf.num_row_groups):
    table = pf.read_row_group(rg, columns=["s1_idx", "cand_code", "n_signals"])
    df = table.to_pandas()
    if df.empty:
        continue
    key = df["s1_idx"].astype(np.int64) * CODE_MULT + df["cand_code"].astype(np.int64)
    df["label"] = key.isin(true_keys).astype(np.int8)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_pos_seen += len(pos)
    n_neg_seen += len(neg)

    # keep all positives found (blocking recall is already <1.0, don't throw
    # away more of the already-scarce true matches by subsampling them)
    pos_parts.append(pos)

    # negatives: sample down per row group so total volume stays manageable
    # (final ratio is enforced below, this just avoids holding everything)
    keep_n = min(len(neg), max(1, int(len(pos) * N_NEGATIVE_PER_POSITIVE * 2)))
    if len(neg) > keep_n:
        neg = neg.sample(n=keep_n, random_state=RANDOM_SEED + rg)
    neg_parts.append(neg)

    if rg % 20 == 0:
        print(f"  row group {rg}/{pf.num_row_groups}: "
              f"pos_seen={n_pos_seen:,} neg_seen={n_neg_seen:,}", flush=True)

positive = pd.concat(pos_parts, ignore_index=True)
negative_pool = pd.concat(neg_parts, ignore_index=True)
print(f"\nTotal positives found in candidate parquet: {len(positive):,} "
      f"(this is the true ceiling -- it equals blocking recall x true pairs)")
print(f"Negative pool collected: {len(negative_pool):,}")

# final negative sample at the desired ratio to the ACTUAL positive count
target_neg = min(len(negative_pool), len(positive) * N_NEGATIVE_PER_POSITIVE)
negative = negative_pool.sample(n=target_neg, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Negatives sampled for training: {len(negative):,} "
      f"(ratio {len(negative) / max(len(positive), 1):.1f}:1)")

train = pd.concat([positive, negative], ignore_index=True)

print("\nDecoding entity IDs and assigning split...")
train["s1_entity_id"] = train["s1_idx"].map(lambda i: s1_ids[int(i)])
train["target_entity_id"] = train["cand_code"].map(lambda c: rid_ids[int(c)])
train["split"] = train["s1_idx"].map(group_split)
train = train.drop(columns=["cand_code"])

# sanity check: this MUST print 0, or the split still leaks by entity
s1_train = set(train.loc[train["split"] == "train", "s1_idx"])
s1_valid = set(train.loc[train["split"] == "valid", "s1_idx"])
overlap = s1_train & s1_valid
print(f"S1 entities present in BOTH splits (must be 0): {len(overlap)}")
assert len(overlap) == 0, "Split leakage detected -- do not proceed to training."

train = train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print("\nSaving...")
train.to_parquet(OUT_FILE, index=False)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("Output:", OUT_FILE)
print("Rows:", len(train))
print("\nLabels:\n", train["label"].value_counts())
print("\nSplits:\n", train["split"].value_counts())
print("\nn_signals by label (sanity check -- both classes should have real, "
      "overlapping distributions now, not a hard 0-vs-nonzero split):")
print(train.groupby("label")["n_signals"].describe())