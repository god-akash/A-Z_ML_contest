# src/diagnose_union_misses.py
import pickle
import random
import re
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

DATASET = r"F:\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
SIG_CACHE = "cache/multi_signal_sample.pkl"
CAP = 300
N_MISS_SAMPLE = 400
N_EXAMPLES = 6
CHUNK = 250_000

SUFFIX_WORDS = {"pvt", "ltd", "private", "limited", "llc", "inc", "corp", "corporation",
                "co", "company", "gmbh", "llp", "plc", "lp"}
TLD_RE = re.compile(r"\.(com|net|org|info|biz|co\.in|in|us|fr)\b")
TOK_RE = re.compile(r"[a-z0-9]+")


def ascii_lower(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def norm_name(s):
    return " ".join(t for t in TOK_RE.findall(TLD_RE.sub("", ascii_lower(s))) if t not in SUFFIX_WORDS)


def num_tokens(s):
    return {t for t in TOK_RE.findall(ascii_lower(s)) if any(c.isdigit() for c in t)}


d = pickle.load(open(SIG_CACHE, "rb"))
kh_sorted, rid_sorted, rid_map = d["kh"], d["rid"], d["rid_map"]
inv = {c: e for e, c in rid_map.items()}
truth = {e: tm for e, tm in d["truth"].items() if tm}          # drops singletons

missed = []       # (s1_id, other_id, in_rid_map)
n_true = 0
for eid, hs in d["s1_keys"]:
    tm = truth.get(eid)
    if not tm:
        continue
    n_true += len(tm)
    if hs:
        h = np.array([x for _, x in hs], dtype=np.int64)
        lo = np.searchsorted(kh_sorted, h, "left")
        df = np.searchsorted(kh_sorted, h, "right") - lo
        cand = set()
        for l, n in zip(lo, df):
            if 1 <= n <= CAP:
                cand.update(rid_sorted[l:l + n].tolist())
        cand_ids = {inv[c] for c in cand}
    else:
        cand_ids = set()
    for x in tm - cand_ids:
        missed.append((eid, x, x in rid_map))

print(f"True pairs: {n_true:,}   missed by union(cap {CAP}): {len(missed):,} "
      f"({len(missed) / n_true:.1%})")
print(f"  missed pairs whose S2/S3 record shares NO key at all with its S1: "
      f"{sum(1 for m in missed if not m[2]):,}\n")

random.seed(5)
pick = random.sample(missed, min(N_MISS_SAMPLE, len(missed)))
need1 = {m[0] for m in pick}
need_o = {m[1] for m in pick}


def collect(path, needed):
    found = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK, keep_default_na=False):
        hit = chunk[chunk["entity_id"].isin(needed)]
        for r in hit.itertuples(index=False):
            found[r.entity_id] = (r.business_name, r.business_address)
        if len(found) == len(needed):
            break
    return found


print("Fetching raw records...")
raw1 = collect(f"{DATASET}/train_source1.tsv", need1)
raw2 = collect(f"{DATASET}/train_source2.tsv", {i for i in need_o if i.startswith("S2-")})
raw3 = collect(f"{DATASET}/train_source3.tsv", {i for i in need_o if i.startswith("S3-")})
raw_o = {**raw2, **raw3}

groups = defaultdict(list)
for eid, oid, in_map in pick:
    a, b = raw1.get(eid), raw_o.get(oid)
    if not a or not b:
        continue
    n1, n2 = norm_name(a[0]), norm_name(b[0])
    a1, a2 = ascii_lower(a[1]), ascii_lower(b[1])
    name_ratio = fuzz.ratio(n1, n2) if n2 else 0
    addr_ratio = fuzz.token_set_ratio(a1, a2) if (a1 and a2) else 0
    if n2 == "":
        g = "nonlatin_name"
    elif n1 == n2:
        g = "same_name"
    elif name_ratio >= 80:
        g = "name_char_close"
    elif addr_ratio >= 85:
        g = "addr_close_only"
    else:
        g = "both_far"
    groups[g].append((eid, oid, in_map, name_ratio, addr_ratio, a, b))

total = sum(len(v) for v in groups.values())
print(f"\nClassified {total} missed pairs\n")
print(f"{'group':16s} {'count':>6s} {'share':>7s} {'med name ratio':>15s} {'med addr ratio':>15s} "
      f"{'addr>=85':>9s} {'shares a key':>13s}")
for g in ["nonlatin_name", "same_name", "name_char_close", "addr_close_only", "both_far"]:
    rows = groups.get(g, [])
    if not rows:
        continue
    print(f"{g:16s} {len(rows):6d} {len(rows) / total:7.1%} "
          f"{np.median([r[3] for r in rows]):15.0f} {np.median([r[4] for r in rows]):15.0f} "
          f"{np.mean([r[4] >= 85 for r in rows]):9.1%} {np.mean([r[2] for r in rows]):13.1%}")

print("\n================ EXAMPLES ================")
for g in ["nonlatin_name", "same_name", "name_char_close", "addr_close_only", "both_far"]:
    print(f"\n--- {g} ---")
    for eid, oid, in_map, nr, ar, a, b in groups.get(g, [])[:N_EXAMPLES]:
        print(f"  S1: {a[0]!r} | {a[1]!r}")
        print(f"  {oid[:2]}: {b[0]!r} | {b[1]!r}")
        print(f"     name_ratio={nr:.0f} addr_ratio={ar:.0f}\n")