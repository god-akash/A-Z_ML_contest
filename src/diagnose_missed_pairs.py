# src/diagnose_missed_pairs.py
import re
import unicodedata
from collections import Counter, defaultdict
import pandas as pd
from rapidfuzz import fuzz

DATASET = r"D:\A-Z_ML_contest\dataset\train"
N_SAMPLE = 3000
CHUNK = 250_000

SUFFIXES = ["pvt ltd", "private limited", "ltd", "llc", "inc", "corp",
            "corporation", "co", "company", "limited", "gmbh", "plc", "llp"]

def normalize(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    tokens = s.split()
    changed = True
    while changed and tokens:
        changed = False
        for suf in SUFFIXES:
            st = suf.split()
            if len(tokens) > len(st) and tokens[-len(st):] == st:
                tokens = tokens[:-len(st)]
                changed = True
    return " ".join(tokens)

def collect(path, needed):
    found = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK,
                             keep_default_na=False):
        hit = chunk[chunk["entity_id"].isin(needed)]
        for r in hit.itertuples(index=False):
            found[r.entity_id] = (r.business_name, r.business_address, r.country)
        if len(found) == len(needed):
            break
    return found

gt = pd.read_csv(f"{DATASET}/train_ground_truth.tsv", sep="\t", dtype=str,
                 keep_default_na=False)
gt = gt[gt["matched_entity_ids"].str.strip() != ""].sample(N_SAMPLE, random_state=7)
gt["lst"] = gt["matched_entity_ids"].apply(
    lambda x: [i.strip() for i in x.split(",") if i.strip()])

need1 = set(gt["source1_entity_id"])
need2 = {i for l in gt["lst"] for i in l if i.startswith("S2-")}
need3 = {i for l in gt["lst"] for i in l if i.startswith("S3-")}
print("Scanning files...")
r1 = collect(f"{DATASET}/train_source1.tsv", need1)
r2 = collect(f"{DATASET}/train_source2.tsv", need2)
r3 = collect(f"{DATASET}/train_source3.tsv", need3)
rec = {**r2, **r3}

counts = Counter()
examples = defaultdict(list)
reach = Counter()
addr_support = Counter()
total_nonexact = 0
total_pairs = 0

def bucket(n1, n2):
    t1, t2 = n1.split(), n2.split()
    if not t1 or not t2:
        return "empty_name"
    if sorted(t1) == sorted(t2):
        return "word_order"
    a, b = set(t1), set(t2)
    if a <= b or b <= a:
        return "token_subset"
    if min(len(n1), len(n2)) >= 3 and (n1.startswith(n2) or n2.startswith(n1)):
        return "prefix"
    if fuzz.ratio(n1, n2) >= 85:
        return "typo_or_close"
    if fuzz.token_set_ratio(n1, n2) >= 80:
        return "partial_overlap"
    if len(t1[0]) >= 3 and t1[0] == t2[0]:
        return "same_first_token"
    return "unrelated_looking"

for row in gt.itertuples(index=False):
    s1 = r1.get(row.source1_entity_id)
    if s1 is None:
        continue
    n1 = normalize(s1[0])
    group = [(eid, rec[eid]) for eid in row.lst if eid in rec]
    norms = {eid: normalize(v[0]) for eid, v in group}
    anchors = [n for n in norms.values() if n == n1]
    for eid, v in group:
        total_pairs += 1
        n2 = norms[eid]
        if n2 == n1:
            continue
        total_nonexact += 1
        b = bucket(n1, n2)
        counts[b] += 1
        if len(examples[b]) < 5:
            examples[b].append((s1[0], v[0], s1[1], v[1]))
        if anchors and any(fuzz.ratio(n2, a) >= 90 for a in anchors):
            reach[b] += 1
        a1, a2 = str(s1[1]).lower(), str(v[1]).lower()
        if a1 and a2 and fuzz.token_set_ratio(a1, a2) >= 90:
            addr_support[b] += 1

print(f"\nTrue pairs: {total_pairs}   non-exact: {total_nonexact} "
      f"({total_nonexact / total_pairs:.1%})\n")
print(f"{'bucket':20s} {'count':>6s} {'%':>6s} {'sibling-reachable':>18s} {'addr>=90':>9s}")
for b, c in counts.most_common():
    print(f"{b:20s} {c:6d} {c / total_nonexact:6.1%} "
          f"{reach[b] / c:18.1%} {addr_support[b] / c:9.1%}")

print("\n=========== EXAMPLES (S1 name | other name | S1 addr | other addr) ===========")
for b, _ in counts.most_common():
    print(f"\n--- {b} ---")
    for e in examples[b]:
        print(f"  {e[0]!r} | {e[1]!r}\n     {e[2]!r} | {e[3]!r}")