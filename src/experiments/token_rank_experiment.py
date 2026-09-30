# src/token_rank_experiment.py
import heapq
import math
import os
import pickle
import time
from collections import defaultdict
import numpy as np
import pandas as pd

DATASET = r"F:\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
CACHE_PATH = "cache/token_index_sample.pkl"
N_SAMPLE = 10_000
CHUNK = 200_000
MAX_DF = 3000
CAPS = [30, 100, 300, 1000, 3000]   # tokens with df above this do not vote
KS = [10, 20, 50, 100]
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


def build_cache():
    t0 = time.time()
    print("Loading S1 sample + ground truth...")
    s1 = pd.read_csv(f"{DATASET}/train_source1.tsv", sep="\t", dtype=str,
                     usecols=COLS, keep_default_na=False)
    s1 = s1.sample(N_SAMPLE, random_state=1).reset_index(drop=True)
    gt = pd.read_csv(f"{DATASET}/train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = {r.source1_entity_id: {x.strip() for x in r.matched_entity_ids.split(",") if x.strip()}
             for r in gt.itertuples(index=False)}

    ex1 = explode_tokens(s1)
    s1_tokens = ex1.groupby("rid")["tok"].apply(list).to_dict()
    wanted_idx = pd.Index(list(set(ex1["tok"])))
    print(f"Sample S1 unique tokens: {len(wanted_idx):,}")

    store, dfc, dead, n_total = None, pd.Series(dtype="float64"), set(), 0
    for src in ["source2", "source3"]:
        reader = pd.read_csv(f"{DATASET}/train_{src}.tsv", sep="\t", dtype=str,
                             usecols=COLS, chunksize=CHUNK, keep_default_na=False)
        for i, chunk in enumerate(reader):
            n_total += len(chunk)
            ex = explode_tokens(chunk)
            ex = ex[ex["tok"].isin(wanted_idx)]
            dfc = dfc.add(ex["tok"].value_counts(), fill_value=0)
            store = ex if store is None else pd.concat([store, ex], ignore_index=True)
            over = set(dfc.index[dfc > MAX_DF]) - dead
            if over:
                dead |= over
                store = store[~store["tok"].isin(over)]
                wanted_idx = pd.Index(list(set(wanted_idx) - over))
            if i % 5 == 0:
                print(f"  {src} chunk {i}: postings={len(store):,} t={time.time() - t0:.0f}s")

    post = store.groupby("tok")["rid"].apply(list).to_dict()
    rid_country = dict(zip(store["rid"], store["country"]))
    df_map = dfc.to_dict()
    s1_rows = [(r.entity_id, r.country, list(set(s1_tokens.get(r.entity_id, []))))
               for r in s1.itertuples(index=False)]
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"post": post, "rid_country": rid_country, "df_map": df_map,
                     "n_total": n_total, "s1_rows": s1_rows, "truth": truth}, f)
    print(f"Cache saved ({time.time() - t0:.0f}s)")


if not os.path.exists(CACHE_PATH):
    build_cache()

with open(CACHE_PATH, "rb") as f:
    d = pickle.load(f)
post, rid_country, df_map = d["post"], d["rid_country"], d["df_map"]
n_total, s1_rows, truth = d["n_total"], d["s1_rows"], d["truth"]
print(f"\nIndex loaded: N(S2+S3)={n_total:,}, S1 sample={len(s1_rows):,}")

print(f"\n{'cap':>5} | {'ceiling':>7} | " +
      " ".join(f"R@{k:<4}" for k in KS) + f" | {'AllFound@50':>11} | {'mean avail':>10}")
for cap in CAPS:
    t0 = time.time()
    n_true = n_with = hits_all = 0
    hits_k = {k: 0 for k in KS}
    allfound50 = 0
    avail = []
    for eid, country, toks in s1_rows:
        sc = defaultdict(float)
        for t in toks:
            df = df_map.get(t, 0)
            if 1 <= df <= cap:
                w = math.log(n_total / df)
                for r in post[t]:
                    sc[r] += w
        avail.append(sum(1 for r in sc if rid_country[r] == country))
        tm = truth.get(eid)
        if not tm:
            continue
        n_with += 1
        n_true += len(tm)
        hits_all += sum(1 for r in tm if r in sc)
        top = heapq.nlargest(max(KS), ((s, r) for r, s in sc.items()
                                       if rid_country[r] == country))
        for k in KS:
            hits_k[k] += len(tm & {r for _, r in top[:k]})
        allfound50 += tm <= {r for _, r in top[:50]}
    print(f"{cap:>5} | {hits_all / n_true:7.4f} | " +
          " ".join(f"{hits_k[k] / n_true:6.4f}" for k in KS) +
          f" | {allfound50 / n_with:11.4f} | {np.mean(avail):10.1f}   ({time.time() - t0:.0f}s)")