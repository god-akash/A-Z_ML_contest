import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

from rapidfuzz.fuzz import ratio, token_set_ratio
from unidecode import unidecode


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, "dataset", "train")
CACHE = os.path.join(ROOT, "cache")

S1_PATH = os.path.join(DATA, "train_source1.tsv")
S2_PATH = os.path.join(DATA, "train_source2.tsv")
S3_PATH = os.path.join(DATA, "train_source3.tsv")
GT_PATH = os.path.join(DATA, "train_ground_truth.tsv")

CAND_PATH = os.path.join(
    CACHE,
    "train_candidate_pairs.parquet"
)

INDEX_PATH = os.path.join(
    CACHE,
    "train_index.pkl"
)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_MISSES = 10000

# Similarity thresholds
TRANS_THRESHOLD = 80
NAME_THRESHOLD = 85
ADDRESS_THRESHOLD = 85


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "pvt", "ltd",
    "private", "limited",
    "llc", "inc",
    "corp", "corporation",
    "co", "company",
    "gmbh", "llp",
    "plc", "lp"
}


def normalize_unicode(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    ).casefold()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_ascii(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text)
    )

    text = text.encode(
        "ascii",
        "ignore"
    ).decode("ascii")

    text = text.casefold()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def transliterate(text):

    if not text:
        return ""

    text = unidecode(str(text))

    text = text.casefold()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def remove_suffixes(text):

    tokens = text.split()

    return " ".join(
        t for t in tokens
        if t not in LEGAL_SUFFIXES
    )


def norm_name(text):

    return remove_suffixes(
        normalize_unicode(text)
    )


def trans_name(text):

    return remove_suffixes(
        transliterate(text)
    )


def norm_address(text):

    return normalize_unicode(text)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TARGETED MISSED-PAIR RECOVERY EXPERIMENT")
print("=" * 70)

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    keep_default_na=False
)

print(
    f"S1 rows: {len(s1):,}"
)


print("\nLoading S2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    keep_default_na=False
)

print(
    f"S2 rows: {len(s2):,}"
)


print("\nLoading S3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    keep_default_na=False
)

print(
    f"S3 rows: {len(s3):,}"
)


# ============================================================
# CREATE FAST LOOKUPS
# ============================================================

print("\nCreating record lookup...")

records = pd.concat(
    [s2, s3],
    ignore_index=True
)

del s2
del s3


record_lookup = {}

for r in records.itertuples(index=False):

    record_lookup[r.entity_id] = (
        r.business_name,
        r.business_address,
        r.country
    )

del records


s1_lookup = {}

for r in s1.itertuples(index=False):

    s1_lookup[r.entity_id] = (
        r.business_name,
        r.business_address,
        r.country
    )

del s1


# ============================================================
# GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


truth = set()

for r in gt.itertuples(index=False):

    matches = str(
        r.matched_entity_ids
    ).strip()

    if not matches:
        continue

    for other in matches.split(","):

        other = other.strip()

        if other:
            truth.add(
                (
                    r.source1_entity_id,
                    other
                )
            )

print(
    f"TRUE PAIRS: {len(truth):,}"
)


# ============================================================
# IMPORTANT:
# ============================================================

print("\nCandidate file uses internal codes.")
print("Loading candidate ID mapping from cache...")


import pickle

with open(INDEX_PATH, "rb") as f:

    cache = pickle.load(f)


rid_ids = cache["rid_ids"]


print(
    f"Candidate record IDs: "
    f"{len(rid_ids):,}"
)


# Reload S1 IDs in original row order

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id"],
    keep_default_na=False
)

s1_ids = s1["entity_id"].tolist()

del s1


# ============================================================
# SECOND PASS — FIND MISSED PAIRS
# ============================================================

print("\nFinding missed TRUE pairs...")

recovered = set()

candidate_file = pq.ParquetFile(
    CAND_PATH
)

total_rows = candidate_file.metadata.num_rows

scanned = 0

for batch in candidate_file.iter_batches(
    batch_size=500_000,
    columns=[
        "s1_idx",
        "cand_code"
    ]
):

    df = batch.to_pandas()

    for row in df.itertuples(index=False):

        sid = s1_ids[row.s1_idx]

        oid = rid_ids[row.cand_code]

        pair = (sid, oid)

        if pair in truth:

            recovered.add(pair)

    scanned += len(df)

    if scanned % 10_000_000 == 0:

        print(
            f"  scanned {scanned:,} / "
            f"{total_rows:,}"
        )


missed = list(
    truth - recovered
)


print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"TRUE PAIRS: {len(truth):,}"
)

print(
    f"RECOVERED: {len(recovered):,}"
)

print(
    f"MISSED: {len(missed):,}"
)

print(
    f"RECALL: {len(recovered) / len(truth):.4%}"
)


# ============================================================
# SAMPLE MISSED PAIRS
# ============================================================

if len(missed) > SAMPLE_MISSES:

    import random

    random.seed(42)

    missed_sample = random.sample(
        missed,
        SAMPLE_MISSES
    )

else:

    missed_sample = missed


print(
    f"\nAnalysing {len(missed_sample):,} missed pairs..."
)


# ============================================================
# RESULTS
# ============================================================

stats = {
    "trans_close": 0,
    "name_close": 0,
    "address_close": 0,

    "trans_and_address": 0,
    "name_and_address": 0,

    "trans_or_name": 0,
    "any_signal": 0,

    "all_far": 0
}


examples = defaultdict(list)


# ============================================================
# ANALYSE
# ============================================================

for i, (sid, oid) in enumerate(
    missed_sample,
    start=1
):

    s1_name_raw, s1_addr_raw, s1_country = (
        s1_lookup[sid]
    )

    o_name_raw, o_addr_raw, o_country = (
        record_lookup[oid]
    )


    # -------------------------
    # NAME
    # -------------------------

    n1 = norm_name(
        s1_name_raw
    )

    n2 = norm_name(
        o_name_raw
    )

    name_score = token_set_ratio(
        n1,
        n2
    ) if n1 and n2 else 0


    # -------------------------
    # TRANSLITERATED NAME
    # -------------------------

    t1 = trans_name(
        s1_name_raw
    )

    t2 = trans_name(
        o_name_raw
    )

    trans_score = token_set_ratio(
        t1,
        t2
    ) if t1 and t2 else 0


    # -------------------------
    # ADDRESS
    # -------------------------

    a1 = norm_address(
        s1_addr_raw
    )

    a2 = norm_address(
        o_addr_raw
    )

    address_score = token_set_ratio(
        a1,
        a2
    ) if a1 and a2 else 0


    trans_close = (
        trans_score >= TRANS_THRESHOLD
    )

    name_close = (
        name_score >= NAME_THRESHOLD
    )

    address_close = (
        address_score >= ADDRESS_THRESHOLD
    )


    # -------------------------
    # COUNTS
    # -------------------------

    if trans_close:
        stats["trans_close"] += 1

    if name_close:
        stats["name_close"] += 1

    if address_close:
        stats["address_close"] += 1

    if trans_close and address_close:
        stats["trans_and_address"] += 1

    if name_close and address_close:
        stats["name_and_address"] += 1

    if trans_close or name_close:
        stats["trans_or_name"] += 1

    if (
        trans_close
        or name_close
        or address_close
    ):
        stats["any_signal"] += 1

    if not (
        trans_close
        or name_close
        or address_close
    ):
        stats["all_far"] += 1


    # -------------------------
    # EXAMPLES
    # -------------------------

    if trans_close and len(
        examples["trans_close"]
    ) < 10:

        examples["trans_close"].append(
            (
                sid,
                oid,
                s1_name_raw,
                o_name_raw,
                trans_score,
                address_score
            )
        )


    if name_close and len(
        examples["name_close"]
    ) < 10:

        examples["name_close"].append(
            (
                sid,
                oid,
                s1_name_raw,
                o_name_raw,
                name_score,
                address_score
            )
        )


    if address_close and len(
        examples["address_close"]
    ) < 10:

        examples["address_close"].append(
            (
                sid,
                oid,
                s1_name_raw,
                o_name_raw,
                name_score,
                address_score
            )
        )


    if i % 1000 == 0:

        print(
            f"  analysed "
            f"{i:,} / {len(missed_sample):,}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TARGETED RECOVERY RESULTS")
print("=" * 70)

print(
    f"Missed pairs analysed: "
    f"{len(missed_sample):,}"
)

for key, value in stats.items():

    print(
        f"{key:22s}: "
        f"{value:6,d} "
        f"({value / len(missed_sample):7.2%})"
    )


# ============================================================
# POTENTIAL RECALL
# ============================================================

print("\n" + "=" * 70)
print("POTENTIAL RECOVERY")
print("=" * 70)

any_signal = stats["any_signal"]

print(
    f"Pairs recoverable by at least "
    f"one tested signal: "
    f"{any_signal:,} / "
    f"{len(missed_sample):,}"
)

print(
    f"Potential missed-pair recovery: "
    f"{any_signal / len(missed_sample):.2%}"
)


# ============================================================
# EXAMPLES
# ============================================================

for category in [
    "trans_close",
    "name_close",
    "address_close"
]:

    print("\n" + "=" * 70)

    print(
        "EXAMPLES:",
        category
    )

    print("=" * 70)

    for x in examples[category]:

        print("\nS1:", x[0])
        print("OTHER:", x[1])
        print("NAME 1:", x[2])
        print("NAME 2:", x[3])
        print("NAME SCORE:", x[4])
        print("ADDRESS SCORE:", x[5])


print("\nDONE.")