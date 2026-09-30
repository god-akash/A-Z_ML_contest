# src/run_inference.py
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from xgboost import XGBClassifier

ROOT = r"D:\A-Z_ML_contest"
DATASET_ROOT = os.path.join(ROOT, "dataset")
CACHE_DIR = os.path.join(ROOT, "cache")
OUTPUT_DIR = os.path.join(ROOT, "output")

SPLIT = "test"
CAND_FILE = os.path.join(CACHE_DIR, f"{SPLIT}_candidate_pairs.parquet")
ID_MAP_FILE = os.path.join(CACHE_DIR, f"{SPLIT}_id_maps.pkl")
MODEL_FILE = os.path.join(CACHE_DIR, "entity_match_xgb.json")

THRESHOLD = 0.93   # chosen from validation F0.5 sweep

FEATURE_COLS = [
    "name_ratio", "name_token_set", "name_partial", "name_exact",
    "name_token_jaccard", "name_len_diff",
    "address_ratio", "address_token_set", "address_partial", "address_exact",
    "address_token_jaccard", "address_len_diff",
    "number_overlap", "number_count_common",
    "country_match", "n_signals",
    "name_missing", "address_missing",
]


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


def load_lookup(source):
    path = os.path.join(DATASET_ROOT, SPLIT, f"{SPLIT}_{source}.tsv")
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "business_name", "business_address", "country"])
    return dict(zip(df["entity_id"],
                    zip(df["business_name"], df["business_address"], df["country"])))


print("=" * 70)
print("RUNNING TEST INFERENCE")
print("=" * 70)
t0 = time.time()

print("\nLoading id maps...")
with open(ID_MAP_FILE, "rb") as f:
    idmaps = pickle.load(f)
s1_ids = idmaps["s1_ids"]      # every test S1 entity, in candidate-generation order
rid_ids = idmaps["rid_ids"]    # every S2/S3 entity referenced by any candidate
print(f"S1 entities: {len(s1_ids):,}   candidate-side entities: {len(rid_ids):,}")

print("\nLoading trained model...")
model = XGBClassifier()
model.load_model(MODEL_FILE)

print("\nLoading full S1/S2/S3 test lookups...")
s1_lookup = load_lookup("source1")
s2_lookup = load_lookup("source2")
s3_lookup = load_lookup("source3")
other_lookup = {**s2_lookup, **s3_lookup}
del s2_lookup, s3_lookup
print(f"Lookups ready: S1={len(s1_lookup):,}  other={len(other_lookup):,}  "
      f"t={time.time() - t0:.0f}s")

all_candidates = {sid: [] for sid in s1_ids}
matched = {sid: [] for sid in s1_ids}

print("\nStreaming candidate parquet, computing features, scoring...")
pf = pq.ParquetFile(CAND_FILE)
n_scored = 0
n_missing_lookup = 0
t1 = time.time()

for rg in range(pf.num_row_groups):
    table = pf.read_row_group(rg, columns=["s1_idx", "cand_code", "n_signals"])
    df = table.to_pandas()
    if df.empty:
        continue

    s1_idx_arr = df["s1_idx"].to_numpy()
    cand_arr = df["cand_code"].to_numpy()
    nsig_arr = df["n_signals"].to_numpy()

    rows = []
    for s1i, ci, nsig in zip(s1_idx_arr, cand_arr, nsig_arr):
        sid = s1_ids[int(s1i)]
        cid = rid_ids[int(ci)]
        s1_rec = s1_lookup.get(sid)
        o_rec = other_lookup.get(cid)
        if s1_rec is None or o_rec is None:
            n_missing_lookup += 1
            continue

        n1, a1 = norm(s1_rec[0]), norm(s1_rec[1])
        n2, a2 = norm(o_rec[0]), norm(o_rec[1])
        c1 = (s1_rec[2] or "").casefold()
        c2 = (o_rec[2] or "").casefold()

        name_tok1, name_tok2 = set(n1.split()), set(n2.split())
        addr_tok1, addr_tok2 = set(a1.split()), set(a2.split())
        num1 = set(re.findall(r"\d+", a1))
        num2 = set(re.findall(r"\d+", a2))

        rows.append((
            sid, cid,
            fuzz.ratio(n1, n2), fuzz.token_set_ratio(n1, n2), fuzz.partial_ratio(n1, n2),
            int(n1 == n2 and n1 != ""), jaccard(name_tok1, name_tok2), abs(len(n1) - len(n2)),
            fuzz.ratio(a1, a2), fuzz.token_set_ratio(a1, a2), fuzz.partial_ratio(a1, a2),
            int(a1 == a2 and a1 != ""), jaccard(addr_tok1, addr_tok2), abs(len(a1) - len(a2)),
            jaccard(num1, num2), len(num1 & num2),
            int(c1 == c2), int(nsig),
            int(not n1 or not n2), int(not a1 or not a2),
        ))

    if not rows:
        continue

    fdf = pd.DataFrame(rows, columns=["s1_id", "cand_id"] + FEATURE_COLS)
    probs = model.predict_proba(fdf[FEATURE_COLS])[:, 1]
    fdf["prob"] = probs

    for sid, cid in zip(fdf["s1_id"], fdf["cand_id"]):
        all_candidates[sid].append(cid)

    hits = fdf.loc[fdf["prob"] >= THRESHOLD]
    for sid, cid in zip(hits["s1_id"], hits["cand_id"]):
        matched[sid].append(cid)

    n_scored += len(fdf)
    if rg % 20 == 0:
        elapsed = time.time() - t1
        rate = n_scored / elapsed if elapsed > 0 else 0
        print(f"  row group {rg}/{pf.num_row_groups}  scored={n_scored:,}  "
              f"({rate:.0f} pairs/s)  t={elapsed:.0f}s", flush=True)

print(f"\nScoring complete. Total pairs scored: {n_scored:,}  "
      f"missing-lookup skips: {n_missing_lookup:,}  t={time.time() - t1:.0f}s")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nWriting candidate_pairs.tsv (full candidate set fed to the model)...")
cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
with open(cand_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_ids:
        cids = sorted(set(all_candidates[sid]))
        f.write(f"{sid}\t{','.join(cids)}\n")

print("Writing matching_results.tsv (final predicted matches)...")
match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
with open(match_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_ids:
        mids = sorted(set(matched[sid]))
        f.write(f"{sid}\t{','.join(mids)}\n")

n_with_match = sum(1 for sid in s1_ids if matched[sid])
n_singleton = len(s1_ids) - n_with_match
print(f"\nS1 entities with >=1 predicted match: {n_with_match:,}")
print(f"S1 entities predicted as singleton (no match): {n_singleton:,}")
print(f"\nSaved: {cand_path}")
print(f"Saved: {match_path}")
print(f"\nTotal time: {time.time() - t0:.0f}s")