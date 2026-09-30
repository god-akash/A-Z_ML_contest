# src/build_features.py
import os
import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = r"D:\A-Z_ML_contest"
TRAIN_PAIRS = os.path.join(ROOT, "cache", "training_pairs.parquet")
S1_FILE = os.path.join(ROOT, "dataset", "train", "train_source1.tsv")
S2_FILE = os.path.join(ROOT, "dataset", "train", "train_source2.tsv")
S3_FILE = os.path.join(ROOT, "dataset", "train", "train_source3.tsv")
OUT_FILE = os.path.join(ROOT, "cache", "training_features.parquet")

CHUNK_SIZE = 250_000
FEATURE_CHUNK = 50_000      # rows processed per feature-computation batch
RANDOM_SEED = 42

# Downsample targets -- keeps split assignment intact, just thins each class
# proportionally within train and within valid separately.
MAX_POS_TRAIN = 1_400_000
MAX_NEG_TRAIN = 5_600_000
MAX_POS_VALID = 150_000
MAX_NEG_VALID = 600_000


def norm(text):
    if pd.isna(text):
        return ""
    text = str(text).casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


print("=" * 70)
print("BUILDING ML FEATURES")
print("=" * 70)

t0 = time.time()
print("\nLoading training pairs...")
pairs = pd.read_parquet(TRAIN_PAIRS)
print(f"Pairs loaded: {len(pairs):,}  t={time.time()-t0:.0f}s")

print("\nDownsampling to a workable size (by split, keeping ratio)...")
parts = []
for split, max_pos, max_neg in [
    ("train", MAX_POS_TRAIN, MAX_NEG_TRAIN),
    ("valid", MAX_POS_VALID, MAX_NEG_VALID),
]:
    sub = pairs[pairs["split"] == split]
    pos = sub[sub["label"] == 1]
    neg = sub[sub["label"] == 0]
    if len(pos) > max_pos:
        pos = pos.sample(n=max_pos, random_state=RANDOM_SEED)
    if len(neg) > max_neg:
        neg = neg.sample(n=max_neg, random_state=RANDOM_SEED)
    parts.append(pos)
    parts.append(neg)
    print(f"  {split}: kept {len(pos):,} positives / {len(neg):,} negatives")

pairs = pd.concat(parts, ignore_index=True)
pairs = pairs.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Downsampled total: {len(pairs):,}  t={time.time()-t0:.0f}s")

print("\nLoading Source 1...")
s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
s1_lookup = s1.set_index("entity_id")[["business_name", "business_address", "country"]]
s1_lookup.columns = ["s1_name", "s1_address", "s1_country"]
print(f"S1 loaded: {len(s1):,}  t={time.time()-t0:.0f}s")

needed_targets = set(pairs["target_entity_id"])
print(f"\nTarget IDs needed: {len(needed_targets):,}")


def load_targets(path, source_name):
    print(f"\nScanning {source_name}...")
    pieces = []
    total = found = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK_SIZE,
                             keep_default_na=False):
        total += len(chunk)
        mask = chunk["entity_id"].isin(needed_targets)
        if mask.any():
            pieces.append(chunk.loc[mask, ["entity_id", "business_name",
                                            "business_address", "country"]])
            found += mask.sum()
        if total % 1_000_000 < CHUNK_SIZE:
            print(f"  scanned={total:,} found={found:,}", flush=True)
    result = (pd.concat(pieces, ignore_index=True) if pieces
             else pd.DataFrame(columns=["entity_id", "business_name",
                                        "business_address", "country"]))
    print(f"{source_name} needed records found: {len(result):,}")
    return result


s2 = load_targets(S2_FILE, "Source 2")
s3 = load_targets(S3_FILE, "Source 3")
targets = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=["entity_id"])
targets_lookup = targets.set_index("entity_id")[["business_name", "business_address", "country"]]
targets_lookup.columns = ["target_name", "target_address", "target_country"]
del s2, s3, targets
print(f"\nTarget lookup ready: {len(targets_lookup):,}  t={time.time()-t0:.0f}s")

print("\nJoining records...")
data = pairs.join(s1_lookup, on="s1_entity_id")
data = data.join(targets_lookup, on="target_entity_id")
missing_s1 = data["s1_name"].isna().sum()
missing_target = data["target_name"].isna().sum()
print(f"Rows with missing S1 data: {missing_s1:,}  "
      f"Rows with missing target data: {missing_target:,}")
if missing_s1 or missing_target:
    print("WARNING: dropping rows with missing joins before feature computation.")
    data = data.dropna(subset=["s1_name", "target_name"]).reset_index(drop=True)
print(f"Joined rows: {len(data):,}  t={time.time()-t0:.0f}s")

print("\nNormalizing text (vectorized)...")
data["s1_name_n"] = data["s1_name"].map(norm)
data["target_name_n"] = data["target_name"].map(norm)
data["s1_address_n"] = data["s1_address"].map(norm)
data["target_address_n"] = data["target_address"].map(norm)
data["s1_country_l"] = data["s1_country"].fillna("").str.casefold()
data["target_country_l"] = data["target_country"].fillna("").str.casefold()

print(f"\nComputing fuzzy features in batches of {FEATURE_CHUNK:,}...")
n = len(data)
feature_rows = []
t1 = time.time()
cols = data[["s1_name_n", "target_name_n", "s1_address_n", "target_address_n",
            "s1_country_l", "target_country_l", "n_signals"]].to_numpy(dtype=object)

for start in range(0, n, FEATURE_CHUNK):
    end = min(start + FEATURE_CHUNK, n)
    batch = cols[start:end]
    for n1, n2, a1, a2, c1, c2, nsig in batch:
        name_tok1, name_tok2 = set(n1.split()), set(n2.split())
        addr_tok1, addr_tok2 = set(a1.split()), set(a2.split())
        num1 = set(re.findall(r"\d+", a1))
        num2 = set(re.findall(r"\d+", a2))
        feature_rows.append((
            fuzz.ratio(n1, n2),
            fuzz.token_set_ratio(n1, n2),
            fuzz.partial_ratio(n1, n2),
            int(n1 == n2 and n1 != ""),
            jaccard(name_tok1, name_tok2),
            abs(len(n1) - len(n2)),
            fuzz.ratio(a1, a2),
            fuzz.token_set_ratio(a1, a2),
            fuzz.partial_ratio(a1, a2),
            int(a1 == a2 and a1 != ""),
            jaccard(addr_tok1, addr_tok2),
            abs(len(a1) - len(a2)),
            jaccard(num1, num2),
            len(num1 & num2),
            int(c1 == c2),
            nsig,
            int(not n1 or not n2),
            int(not a1 or not a2),
        ))
    if (start // FEATURE_CHUNK) % 10 == 0:
        elapsed = time.time() - t1
        rate = end / elapsed
        remaining = (n - end) / rate if rate > 0 else 0
        print(f"  {end:,}/{n:,} rows  ({rate:.0f} rows/s, "
              f"~{remaining/60:.1f} min remaining)", flush=True)

FEATURE_COLS = [
    "name_ratio", "name_token_set", "name_partial", "name_exact",
    "name_token_jaccard", "name_len_diff",
    "address_ratio", "address_token_set", "address_partial", "address_exact",
    "address_token_jaccard", "address_len_diff",
    "number_overlap", "number_count_common",
    "country_match", "n_signals",
    "name_missing", "address_missing",
]
feature_df = pd.DataFrame(feature_rows, columns=FEATURE_COLS)
print(f"\nFeature computation done. t={time.time()-t0:.0f}s")

result = pd.concat([
    data[["s1_idx", "s1_entity_id", "target_entity_id", "label", "split"]].reset_index(drop=True),
    feature_df.reset_index(drop=True),
], axis=1)

print("\nSaving...")
result.to_parquet(OUT_FILE, index=False)

print("\n" + "=" * 70)
print("FEATURE BUILD COMPLETE")
print("=" * 70)
print("Output:", OUT_FILE)
print("Rows:", len(result))
print("\nLabel distribution:\n", result["label"].value_counts())
print("\nSplit distribution:\n", result["split"].value_counts())
print(f"\nTotal time: {time.time()-t0:.0f}s")