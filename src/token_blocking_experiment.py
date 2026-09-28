# src/token_blocking_experiment.py
import time
from collections import Counter
import numpy as np
import pandas as pd

DATASET = r"F:\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
N_SAMPLE = 10_000
CHUNK = 200_000
MAX_DF = 3000          # tokens more common than this are never indexed
# (k rarest tokens, df cap, min shared tokens)
CONFIGS = [
    (3, 300, 1),
    (5, 1000, 1),
    (5, 1000, 2),
    (8, 3000, 1),
    (8, 3000, 2),
    (8, 3000, 3),
]
COLS = ["entity_id", "business_name", "business_address", "country"]


def token_lists(names, addrs):
    text = names + " " + addrs
    text = (text.str.normalize("NFKD")
                .str.encode("ascii", "ignore").str.decode("ascii")
                .str.lower()
                .str.replace(r"[^a-z0-9]+", " ", regex=True))
    return text.str.split()


def explode_tokens(df):
    tl = token_lists(df["business_name"], df["business_address"])
    ex = pd.DataFrame({"rid": df["entity_id"].values,
                       "country": df["country"].values,
                       "tok": tl.values}).explode("tok")
    ex = ex.dropna(subset=["tok"])
    ex = ex[ex["tok"].str.len() >= 2]
    return ex.drop_duplicates(["rid", "tok"])


t0 = time.time()
print("Loading S1 sample + ground truth...")
s1 = pd.read_csv(f"{DATASET}/train_source1.tsv", sep="\t", dtype=str,
                 usecols=COLS, keep_default_na=False)
s1 = s1.sample(N_SAMPLE, random_state=1).reset_index(drop=True)
gt = pd.read_csv(f"{DATASET}/train_ground_truth.tsv", sep="\t", dtype=str,
                 keep_default_na=False)
sample_ids = set(s1["entity_id"])
gt = gt[gt["source1_entity_id"].isin(sample_ids)]
truth = {r.source1_entity_id: {x.strip() for x in r.matched_entity_ids.split(",") if x.strip()}
         for r in gt.itertuples(index=False)}

ex1 = explode_tokens(s1)
s1_tokens = ex1.groupby("rid")["tok"].apply(list).to_dict()
wanted = set(ex1["tok"])
print(f"Sample S1 unique tokens: {len(wanted):,}")

# ---- single pass over S2 and S3: count df + keep postings of non-common tokens
store = None
dfc = pd.Series(dtype="float64")
dead = set()
wanted_idx = pd.Index(list(wanted))
for src in ["source2", "source3"]:
    reader = pd.read_csv(f"{DATASET}/train_{src}.tsv", sep="\t", dtype=str,
                         usecols=COLS, chunksize=CHUNK, keep_default_na=False)
    for i, chunk in enumerate(reader):
        ex = explode_tokens(chunk)
        ex = ex[ex["tok"].isin(wanted_idx)]
        dfc = dfc.add(ex["tok"].value_counts(), fill_value=0)
        store = ex if store is None else pd.concat([store, ex], ignore_index=True)
        over = set(dfc.index[dfc > MAX_DF]) - dead
        if over:
            dead |= over
            store = store[~store["tok"].isin(over)]
            wanted_idx = pd.Index(list(set(wanted_idx) - over))
        print(f"  {src} chunk {i}: postings={len(store):,} "
              f"dead_tokens={len(dead):,} t={time.time() - t0:.0f}s")

post = store.groupby("tok")["rid"].apply(list).to_dict()
rid_country = dict(zip(store["rid"], store["country"]))
df_map = dfc.to_dict()

# ---- per-S1 tokens sorted from rarest to most common (only tokens that exist in S2/S3)
s1_rows = []
for r in s1.itertuples(index=False):
    toks = [t for t in set(s1_tokens.get(r.entity_id, []))
            if 1 <= df_map.get(t, 0) <= MAX_DF]
    toks.sort(key=lambda t: df_map[t])
    s1_rows.append((r.entity_id, r.country, toks))

# ---- evaluate
print("\nEvaluating configs...")
print(f"{'k':>2} {'cap':>5} {'min':>3} | {'recall':>7} {'S1 all-found':>12} | "
      f"{'mean':>6} {'p95':>6} {'p99':>6} {'max':>6} | {'singleton mean':>14}")
for (k, cap, m) in CONFIGS:
    n_true = n_hit = n_with = n_all = 0
    counts, single_counts = [], []
    for eid, country, toks in s1_rows:
        chosen = [t for t in toks if df_map[t] <= cap][:k]
        cnt = Counter()
        for t in chosen:
            cnt.update(post[t])
        cands = {r for r, c in cnt.items() if c >= m and rid_country[r] == country}
        counts.append(len(cands))
        tm = truth.get(eid, set())
        if tm:
            hits = tm & cands
            n_true += len(tm)
            n_hit += len(hits)
            n_with += 1
            n_all += (hits == tm)
        else:
            single_counts.append(len(cands))
    c = np.array(counts)
    print(f"{k:>2} {cap:>5} {m:>3} | {n_hit / n_true:7.4f} {n_all / n_with:12.4f} | "
          f"{c.mean():6.1f} {np.percentile(c, 95):6.0f} {np.percentile(c, 99):6.0f} "
          f"{c.max():6d} | {np.mean(single_counts):14.1f}")
print(f"\nDone in {time.time() - t0:.0f}s")