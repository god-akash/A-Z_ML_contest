import os
import re
import time
import unicodedata
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio


# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "dataset", "train")

S1_PATH = os.path.join(TRAIN, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN, "train_source3.tsv")
GT_PATH = os.path.join(TRAIN, "train_ground_truth.tsv")

N_SAMPLE = 5000
CHUNK = 250_000

# Maximum records allowed behind one blocking key.
MAX_BLOCK = 300

# Candidates retained from each retrieval route.
SIGNAL_CAP = 100

# Final candidates.
TOP_K = 300

# Address tokens that are too generic to be useful.
ADDRESS_STOP = {
    "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "drive", "dr", "park",
    "building", "block", "floor", "unit",
    "city", "district", "state", "country",
    "private", "limited", "pvt", "ltd",
    "llc", "llp", "inc", "company", "corporation",
}


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "private", "limited", "pvt", "ltd",
    "llc", "llp", "inc", "incorporated",
    "corporation", "corp", "company", "co", "plc", "gmbh"
}


def norm(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def name_norm(text):
    x = norm(text)

    if not x:
        return ""

    toks = re.findall(r"\w+", x, flags=re.UNICODE)

    toks = [
        t for t in toks
        if t not in LEGAL_SUFFIXES
    ]

    return " ".join(toks)


def ascii_norm(text):
    if not text:
        return ""

    x = unicodedata.normalize("NFKD", text)

    x = x.encode(
        "ascii", "ignore"
    ).decode("ascii")

    x = x.casefold()

    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def toks(text):
    if not text:
        return []

    return re.findall(
        r"\w+",
        text,
        flags=re.UNICODE
    )


def numbers(text):
    if not text:
        return []

    return re.findall(
        r"\d+[a-zA-Z]?",
        text
    )


# ============================================================
# NAME SIGNATURES
# ============================================================

def name_signatures(name):
    """
    Several cheap signatures.

    We intentionally use several routes because no single
    signature catches every noisy name.
    """

    ts = toks(name)

    if not ts:
        return set()

    result = set()

    # --------------------------------------------------------
    # First meaningful token
    # --------------------------------------------------------

    if len(ts[0]) >= 3:
        result.add(
            "FIRST:" + ts[0][:5]
        )

    # --------------------------------------------------------
    # Last meaningful token
    # --------------------------------------------------------

    if len(ts[-1]) >= 3:
        result.add(
            "LAST:" + ts[-1][:5]
        )

    # --------------------------------------------------------
    # First two tokens
    # --------------------------------------------------------

    if len(ts) >= 2:
        a = ts[0][:4]
        b = ts[1][:4]

        result.add(
            "PAIR:" + min(a, b) + "|" + max(a, b)
        )

    # --------------------------------------------------------
    # Sorted token signature
    # --------------------------------------------------------

    meaningful = [
        t for t in ts
        if len(t) >= 3
    ]

    if meaningful:

        short = sorted(
            t[:4] for t in meaningful
        )

        result.add(
            "TOK:" + "|".join(short[:4])
        )

    # --------------------------------------------------------
    # ASCII auxiliary signature
    # --------------------------------------------------------

    ax = ascii_norm(name)

    if ax:

        ats = toks(ax)

        if ats:

            result.add(
                "ASCII_FIRST:" + ats[0][:5]
            )

            if len(ats) >= 2:

                a = ats[0][:4]
                b = ats[1][:4]

                result.add(
                    "ASCII_PAIR:"
                    + min(a, b)
                    + "|"
                    + max(a, b)
                )

    return result


# ============================================================
# ADDRESS SIGNATURES
# ============================================================

def address_signatures(address):

    ts = toks(address)

    if not ts:
        return set()

    result = set()

    nums = numbers(address)

    # --------------------------------------------------------
    # House/building numbers
    # --------------------------------------------------------

    for n in nums:

        result.add(
            "NUM:" + n
        )

    # --------------------------------------------------------
    # Useful textual address tokens
    # --------------------------------------------------------

    useful = []

    for t in ts:

        if any(c.isdigit() for c in t):
            useful.append(t)
            continue

        if len(t) < 4:
            continue

        if t in ADDRESS_STOP:
            continue

        useful.append(t)

    # --------------------------------------------------------
    # Individual useful tokens
    # --------------------------------------------------------

    for t in useful:

        result.add(
            "ADDR:" + t[:8]
        )

    # --------------------------------------------------------
    # First useful address token
    # --------------------------------------------------------

    if useful:

        result.add(
            "ADDR_FIRST:" + useful[0][:6]
        )

    # --------------------------------------------------------
    # Two-token combinations
    # --------------------------------------------------------

    if len(useful) >= 2:

        a = useful[0][:5]
        b = useful[1][:5]

        result.add(
            "ADDR_PAIR:"
            + min(a, b)
            + "|"
            + max(a, b)
        )

    # --------------------------------------------------------
    # Postal-like numeric token
    # --------------------------------------------------------

    for n in nums:

        if len(n) >= 5:

            result.add(
                "POST:" + n
            )

    return result


# ============================================================
# LOAD SAMPLE
# ============================================================

print("Loading S1 + ground truth...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

gt = gt.set_index(
    "source1_entity_id"
)


# ------------------------------------------------------------
# Select S1 with matches
# ------------------------------------------------------------

matched_ids = []

for sid, row in gt.iterrows():

    x = row["matched_entity_ids"]

    if pd.isna(x):
        continue

    x = str(x).strip()

    if x:
        matched_ids.append(sid)


rng = np.random.default_rng(42)

sample_ids = rng.choice(
    matched_ids,
    size=min(N_SAMPLE, len(matched_ids)),
    replace=False
)

sample_ids = set(sample_ids)

s1_sample = s1[
    s1["entity_id"].isin(sample_ids)
].copy()

print(
    f"S1 sample={len(s1_sample):,}"
)


# ============================================================
# PREPARE S1
# ============================================================

s1_info = {}

all_keys = set()

for row in s1_sample.itertuples(index=False):

    sid = row.entity_id

    name = name_norm(
        row.business_name
    )

    address = norm(
        row.business_address
    )

    country = str(
        row.country
    ).casefold()

    nk = name_signatures(name)
    ak = address_signatures(address)

    s1_info[sid] = {
        "country": country,
        "name": name,
        "address": address,
        "name_keys": nk,
        "address_keys": ak,
    }

    all_keys.update(nk)
    all_keys.update(ak)

print(
    f"S1 blocking keys={len(all_keys):,}"
)


# ============================================================
# BUILD INDEX
# ============================================================

# One pass only.
#
# We do not first calculate global frequencies.
# Instead, each posting list is capped.
#
# This is much faster than the previous script.

index = defaultdict(list)

start = time.time()

for source_name, path in [
    ("source2", S2_PATH),
    ("source3", S3_PATH),
]:

    for chunk_no, df in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
            chunksize=CHUNK,
            keep_default_na=False,
        )
    ):

        # ----------------------------------------------------
        # Vectorized normalization
        # ----------------------------------------------------

        names = (
            df["business_name"]
            .map(name_norm)
        )

        addresses = (
            df["business_address"]
            .map(norm)
        )

        countries = (
            df["country"]
            .fillna("")
            .str.casefold()
        )

        ids = df["entity_id"].tolist()

        # ----------------------------------------------------
        # Process rows
        #
        # This is still Python-level, but only once and with
        # fewer expensive operations than the previous version.
        # ----------------------------------------------------

        for i in range(len(df)):

            eid = ids[i]

            country = countries.iloc[i]

            name = names.iloc[i]
            address = addresses.iloc[i]

            nk = name_signatures(name)
            ak = address_signatures(address)

            # Country is part of every key.
            keys = (
                ("N|" + country + "|" + k)
                for k in nk
            )

            for key in keys:

                if len(index[key]) < MAX_BLOCK:
                    index[key].append(eid)

            keys = (
                ("A|" + country + "|" + k)
                for k in ak
            )

            for key in keys:

                if len(index[key]) < MAX_BLOCK:
                    index[key].append(eid)

        if chunk_no % 5 == 0:

            print(
                f"{source_name} chunk {chunk_no}: "
                f"index_keys={len(index):,} "
                f"t={time.time()-start:.0f}s"
            )


print(
    f"Index complete: {len(index):,} keys"
)

print(
    f"Index time: {time.time()-start:.0f}s"
)


# ============================================================
# GROUND TRUTH
# ============================================================

truth = {}

for sid in sample_ids:

    if sid not in gt.index:
        continue

    value = gt.loc[
        sid,
        "matched_entity_ids"
    ]

    if pd.isna(value):
        truth[sid] = set()
        continue

    value = str(value).strip()

    if not value:

        truth[sid] = set()

    else:

        truth[sid] = set(
            x.strip()
            for x in value.split(",")
            if x.strip()
        )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

print("\nEvaluating...")


def get_candidates(info):

    votes = Counter()

    country = info["country"]

    # --------------------------------------------------------
    # NAME ROUTES
    # --------------------------------------------------------

    for k in info["name_keys"]:

        key = (
            "N|"
            + country
            + "|"
            + k
        )

        for eid in index.get(key, []):

            votes[eid] += 1

    # --------------------------------------------------------
    # ADDRESS ROUTES
    # --------------------------------------------------------

    for k in info["address_keys"]:

        key = (
            "A|"
            + country
            + "|"
            + k
        )

        for eid in index.get(key, []):

            votes[eid] += 1

    # --------------------------------------------------------
    # Keep candidates with most blocking evidence.
    #
    # IMPORTANT:
    # We do NOT require 2 votes.
    # A candidate with one signal can still be a true match.
    # --------------------------------------------------------

    return [
        eid
        for eid, votes_count
        in votes.most_common(TOP_K)
    ]


# ============================================================
# EVALUATION
# ============================================================

total_true = 0
total_found = 0

recalls = []
candidate_counts = []

all_found = 0

for idx, (_, row) in enumerate(
    s1_sample.iterrows()
):

    sid = row["entity_id"]

    true_ids = truth.get(
        sid,
        set()
    )

    if not true_ids:
        continue

    candidates = get_candidates(
        s1_info[sid]
    )

    candidate_set = set(candidates)

    found = true_ids & candidate_set

    total_true += len(true_ids)
    total_found += len(found)

    recalls.append(
        len(found) / len(true_ids)
    )

    candidate_counts.append(
        len(candidates)
    )

    if len(found) == len(true_ids):
        all_found += 1

    if idx % 500 == 0:

        print(
            f"evaluated={idx:,} "
            f"mean_candidates="
            f"{np.mean(candidate_counts):.1f}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FAST BLOCKING V2 RESULTS")
print("=" * 70)

print(
    f"True pairs: {total_true:,}"
)

print(
    f"Recovered: {total_found:,}"
)

print(
    f"Candidate recall: "
    f"{total_found / total_true:.4f}"
)

print(
    f"All matches found: "
    f"{all_found / len(s1_sample):.4f}"
)

print(
    f"Mean candidates: "
    f"{np.mean(candidate_counts):.1f}"
)

print(
    f"P95 candidates: "
    f"{np.percentile(candidate_counts, 95):.0f}"
)

print(
    f"P99 candidates: "
    f"{np.percentile(candidate_counts, 99):.0f}"
)

print(
    f"Max candidates: "
    f"{np.max(candidate_counts):,}"
)

print(
    f"Median recall per S1: "
    f"{np.median(recalls):.4f}"
)

print(
    f"Minimum recall per S1: "
    f"{np.min(recalls):.4f}"
)