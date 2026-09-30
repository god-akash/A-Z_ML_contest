# src/inspect_true_matches.py
import re
import unicodedata
import random
import pandas as pd
from rapidfuzz import fuzz

DATASET = r"F:\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
N_SAMPLE = 300
CHUNK = 200_000
random.seed(42)

def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # strip common legal suffixes (extend once we see real data)
    suffixes = [
        "pvt ltd", "private limited", "ltd", "llc", "inc", "corp",
        "corporation", "co", "company", "limited", "gmbh", "plc", "llp"
    ]
    tokens = s.split()
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    # handle multi-word suffixes crudely
    joined = " ".join(tokens)
    for suf in suffixes:
        if joined.endswith(" " + suf):
            joined = joined[: -(len(suf) + 1)]
    return re.sub(r"\s+", " ", joined).strip()

print("Loading ground truth...")
gt = pd.read_csv(
    f"{DATASET}/train_ground_truth.tsv", sep="\t", dtype=str
).fillna("")
gt = gt[gt["matched_entity_ids"].str.strip() != ""]

sample = gt.sample(n=N_SAMPLE, random_state=42).copy()
sample["matched_list"] = sample["matched_entity_ids"].apply(
    lambda x: [i.strip() for i in x.split(",") if i.strip()]
)

need_s2, need_s3 = set(), set()
for lst in sample["matched_list"]:
    for eid in lst:
        if eid.startswith("S2-"):
            need_s2.add(eid)
        elif eid.startswith("S3-"):
            need_s3.add(eid)

need_s1 = set(sample["source1_entity_id"])

print(f"Need {len(need_s1)} S1, {len(need_s2)} S2, {len(need_s3)} S3 records")

def collect(path, needed_ids):
    found = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK):
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        for _, row in hit.iterrows():
            found[row["entity_id"]] = row.to_dict()
        if len(found) == len(needed_ids):
            break
    return found

print("Scanning source1...")
s1_rows = collect(f"{DATASET}/train_source1.tsv", need_s1)
print("Scanning source2...")
s2_rows = collect(f"{DATASET}/train_source2.tsv", need_s2)
print("Scanning source3...")
s3_rows = collect(f"{DATASET}/train_source3.tsv", need_s3)

def get_row(eid):
    if eid.startswith("S2-"):
        return s2_rows.get(eid)
    if eid.startswith("S3-"):
        return s3_rows.get(eid)
    return None

records = []
for _, row in sample.iterrows():
    s1 = s1_rows.get(row["source1_entity_id"])
    if s1 is None:
        continue
    for eid in row["matched_list"]:
        other = get_row(eid)
        if other is None:
            continue
        n1, n2 = s1["business_name"], other["business_name"]
        a1, a2 = s1.get("business_address", ""), other.get("business_address", "")
        norm1, norm2 = normalize(n1), normalize(n2)
        records.append({
            "s1_id": row["source1_entity_id"],
            "other_id": eid,
            "s1_name": n1,
            "other_name": n2,
            "raw_exact": str(n1).strip().lower() == str(n2).strip().lower(),
            "norm1": norm1,
            "norm2": norm2,
            "norm_exact": norm1 == norm2,
            "name_ratio": fuzz.ratio(str(n1), str(n2)),
            "name_token_sort": fuzz.token_sort_ratio(str(n1), str(n2)),
            "name_token_set": fuzz.token_set_ratio(str(n1), str(n2)),
            "addr_token_set": fuzz.token_set_ratio(str(a1), str(a2)),
            "country_match": s1["country"] == other["country"],
            "s1_addr": a1,
            "other_addr": a2,
            "s1_country": s1["country"],
            "other_country": other["country"],
        })

df = pd.DataFrame(records)
print(f"\nTotal true pairs inspected: {len(df)}")
print(f"Raw case-insensitive exact match: {df['raw_exact'].mean():.2%}")
print(f"Normalized exact match:           {df['norm_exact'].mean():.2%}")
print(f"Country match:                    {df['country_match'].mean():.2%}")
print("\nname_token_set_ratio distribution:")
print(df["name_token_set"].describe())
print("\naddr_token_set_ratio distribution:")
print(df["addr_token_set"].describe())

out_path = "output/true_match_inspection.csv"
import os
os.makedirs("output", exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} example rows to {out_path} — open it and eyeball 30-50 rows.")