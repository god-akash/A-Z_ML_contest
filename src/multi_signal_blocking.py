# src/multi_signal_blocking.py
import os
import pickle
import re
import time
import unicodedata
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

DATASET = r"F:\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
TOKEN_CACHE = "cache/token_index_sample.pkl"     # made by token_rank_experiment.py
SIG_CACHE = "cache/multi_signal_sample.pkl"
N_S1 = 5000
CHUNK = 200_000
KEY_CAP = 300                                    # keys shared by more records are dropped
CAPS = [20, 100, 300]
SIGNALS = ["nx", "ns", "nc", "np", "ap", "an", "xn"]
SIG_ID = {s: i for i, s in enumerate(SIGNALS)}
COLS = ["entity_id", "business_name", "business_address", "country"]

SUFFIX_WORDS = {"pvt", "ltd", "private", "limited", "llc", "inc", "corp", "corporation",
                "co", "company", "gmbh", "llp", "plc", "lp"}
TLD_RE = re.compile(r"\.(com|net|org|info|biz|co\.in|in|us|fr)\b")
TOK_RE = re.compile(r"[a-z0-9]+")


def ascii_lower(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def keys_for(name, addr, country, vocab):
    c = country.strip().lower()
    ntoks = [t for t in TOK_RE.findall(TLD_RE.sub("", ascii_lower(name))) if t not in SUFFIX_WORDS]
    atoks = TOK_RE.findall(ascii_lower(addr))
    keys = []
    if ntoks:
        keys.append(f"nx|{c}|" + " ".join(ntoks))
        keys.append(f"ns|{c}|" + " ".join(sorted(set(ntoks))))
        nc = "".join(ntoks)
        if len(nc) >= 6:
            keys.append(f"nc|{c}|{nc}")
    nv = sorted({t for t in ntoks if len(t) >= 3 and t in vocab})[:8]
    aa = sorted({t for t in atoks if t.isalpha() and len(t) >= 4 and t in vocab})[:8]
    an = sorted({t for t in atoks if len(t) >= 2 and any(ch.isdigit() for ch in t) and t in vocab})[:4]
    for a, b in combinations(nv, 2):
        keys.append(f"np|{c}|{a}|{b}")
    for a, b in combinations(aa, 2):
        keys.append(f"ap|{c}|{a}|{b}")
    for a in an:
        for b in aa:
            keys.append(f"an|{c}|{a}|{b}")
    for a in nv:
        for b in an:
            keys.append(f"xn|{c}|{a}|{b}")
    return keys


def build_cache():
    t0 = time.time()
    tc = pickle.load(open(TOKEN_CACHE, "rb"))
    vocab = {t for t, df in tc["df_map"].items() if 1 <= df <= 3000}
    s1_ids = [r[0] for r in tc["s1_rows"][:N_S1]]
    truth = {e: tc["truth"][e] for e in s1_ids if e in tc["truth"]}
    print(f"vocab={len(vocab):,}  S1 sample={len(s1_ids):,}  with matches={len(truth):,}")

    s1 = pd.read_csv(f"{DATASET}/train_source1.tsv", sep="\t", dtype=str,
                     usecols=COLS, keep_default_na=False)
    s1 = s1[s1["entity_id"].isin(set(s1_ids))]
    s1_keys, key_set = [], set()
    for r in s1.itertuples(index=False):
        hs = []
        for k in keys_for(r.business_name, r.business_address, r.country, vocab):
            h = hash(k)      # Python hash: consistent inside this run only (fine for an experiment)
            hs.append((SIG_ID[k[:2]], h))
        key_set.update(h for _, h in hs)
        s1_keys.append((r.entity_id, hs))
    print(f"S1 keys: {len(key_set):,}")

    rid_map, rid_ids = {}, []
    store, dead = None, set()
    for src in ["source2", "source3"]:
        reader = pd.read_csv(f"{DATASET}/train_{src}.tsv", sep="\t", dtype=str,
                             usecols=COLS, chunksize=CHUNK, keep_default_na=False)
        for i, chunk in enumerate(reader):
            kh_l, rid_l, sig_l = [], [], []
            for r in chunk.itertuples(index=False):
                for k in keys_for(r.business_name, r.business_address, r.country, vocab):
                    h = hash(k)
                    if h in key_set:
                        code = rid_map.get(r.entity_id)
                        if code is None:
                            code = len(rid_ids)
                            rid_map[r.entity_id] = code
                            rid_ids.append(r.entity_id)
                        kh_l.append(h)
                        rid_l.append(code)
                        sig_l.append(SIG_ID[k[:2]])
            new = pd.DataFrame({"kh": np.array(kh_l, dtype=np.int64),
                                "rid": np.array(rid_l, dtype=np.int32),
                                "sig": np.array(sig_l, dtype=np.int8)})
            if dead:
                new = new[~new["kh"].isin(dead)]
            store = new if store is None else pd.concat([store, new], ignore_index=True)
            counts = store["kh"].value_counts()
            over = counts.index[counts > KEY_CAP]
            if len(over):
                dead.update(over.tolist())
                store = store[~store["kh"].isin(over)]
            if i % 5 == 0:
                print(f"  {src} chunk {i}: postings={len(store):,} dead_keys={len(dead):,} "
                      f"t={time.time() - t0:.0f}s", flush=True)

    order = np.argsort(store["kh"].values, kind="stable")
    os.makedirs("cache", exist_ok=True)
    with open(SIG_CACHE, "wb") as f:
        pickle.dump({"kh": store["kh"].values[order], "rid": store["rid"].values[order],
                     "rid_map": rid_map, "s1_keys": s1_keys, "truth": truth}, f)
    print(f"Cache saved ({time.time() - t0:.0f}s)")


if not os.path.exists(SIG_CACHE):
    build_cache()

d = pickle.load(open(SIG_CACHE, "rb"))
kh_sorted, rid_sorted, rid_map = d["kh"], d["rid"], d["rid_map"]
truth = d["truth"]

s1_eval = []
for eid, hs in d["s1_keys"]:
    if not hs:
        s1_eval.append((eid, np.array([], int), np.array([], int), np.array([], int)))
        continue
    sg = np.array([s for s, _ in hs])
    h = np.array([x for _, x in hs], dtype=np.int64)
    lo = np.searchsorted(kh_sorted, h, "left")
    df = np.searchsorted(kh_sorted, h, "right") - lo
    s1_eval.append((eid, sg, lo, df))

truth_codes = {e: ({rid_map[x] for x in tm if x in rid_map}, len(tm)) for e, tm in truth.items()}
n_true = sum(n for _, n in truth_codes.values())
n_with = len(truth_codes)
print(f"\nS1 evaluated={len(s1_eval):,}  with matches={n_with:,}  true pairs={n_true:,}")
print(f"Pairs whose S2/S3 record shares NO key with any sampled S1 signal is measured by the union ceiling below.\n")

# ---------------- per-signal recall (recall / mean candidates)
print("PER SIGNAL  recall (mean candidates per S1)")
print(f"{'signal':>6} | " + " | ".join(f"cap {c:>3}" .center(16) for c in CAPS))
for si, sname in enumerate(SIGNALS):
    cells = []
    for cap in CAPS:
        hit, sizes = 0, []
        for eid, sg, lo, df in s1_eval:
            sel = np.where((sg == si) & (df >= 1) & (df <= cap))[0]
            cand = set()
            for j in sel:
                cand.update(rid_sorted[lo[j]:lo[j] + df[j]].tolist())
            sizes.append(len(cand))
            if eid in truth_codes:
                hit += len(truth_codes[eid][0] & cand)
        cells.append(f"{hit / n_true:.3f} ({np.mean(sizes):5.1f})".center(16))
    print(f"{sname:>6} | " + " | ".join(cells))

# ---------------- union + agreement between signals
print("\nUNION OF ALL 7 SIGNALS")
for cap in [100, 300]:
    hit_u = 0
    hit_v = {2: 0, 3: 0}
    hit_top = {10: 0, 20: 0, 50: 0}
    all20 = 0
    size_u, size_v2, size_v3, size_single = [], [], [], []
    for eid, sg, lo, df in s1_eval:
        sel = np.where((df >= 1) & (df <= cap))[0]
        score, mask = defaultdict(float), defaultdict(int)
        for j in sel:
            w = 1.0 / df[j]
            bit = 1 << int(sg[j])
            for r in rid_sorted[lo[j]:lo[j] + df[j]].tolist():
                score[r] += w
                mask[r] |= bit
        votes = {r: bin(m).count("1") for r, m in mask.items()}
        v2 = {r for r, v in votes.items() if v >= 2}
        v3 = {r for r, v in votes.items() if v >= 3}
        size_u.append(len(score)); size_v2.append(len(v2)); size_v3.append(len(v3))
        tc_, _ = truth_codes.get(eid, (None, 0))
        if tc_ is None:
            size_single.append(len(v2))
            continue
        hit_u += len(tc_ & set(score))
        hit_v[2] += len(tc_ & v2)
        hit_v[3] += len(tc_ & v3)
        ranked = sorted(score, key=lambda r: (votes[r], score[r]), reverse=True)
        for k in hit_top:
            hit_top[k] += len(tc_ & set(ranked[:k]))
        all20 += tc_ <= set(ranked[:20]) and len(tc_) == truth_codes[eid][1]
    su = np.array(size_u)
    print(f"\n key cap {cap}:")
    print(f"  union            recall {hit_u / n_true:.4f} | mean cands {su.mean():7.1f}  p95 {np.percentile(su, 95):6.0f}  p99 {np.percentile(su, 99):6.0f}")
    print(f"  >=2 signals      recall {hit_v[2] / n_true:.4f} | mean cands {np.mean(size_v2):7.1f} | singleton mean {np.mean(size_single):5.1f}")
    print(f"  >=3 signals      recall {hit_v[3] / n_true:.4f} | mean cands {np.mean(size_v3):7.1f}")
    print(f"  top10/20/50 by (votes, score) recall: "
          f"{hit_top[10] / n_true:.4f} / {hit_top[20] / n_true:.4f} / {hit_top[50] / n_true:.4f}"
          f" | all matches in top20 for {all20 / n_with:.1%} of S1")