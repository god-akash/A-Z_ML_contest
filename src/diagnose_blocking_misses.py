import os
import pickle
import random
import re
import unicodedata
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio, token_set_ratio


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_ROOT = os.path.join(
    ROOT,
    "dataset",
)

CACHE_DIR = os.path.join(
    ROOT,
    "cache",
)

CANDIDATE_FILE = os.path.join(
    CACHE_DIR,
    "train_candidate_pairs.parquet",
)

INDEX_FILE = os.path.join(
    CACHE_DIR,
    "train_index.pkl",
)

S1_FILE = os.path.join(
    DATASET_ROOT,
    "train",
    "train_source1.tsv",
)

S2_FILE = os.path.join(
    DATASET_ROOT,
    "train",
    "train_source2.tsv",
)

S3_FILE = os.path.join(
    DATASET_ROOT,
    "train",
    "train_source3.tsv",
)

GT_FILE = os.path.join(
    DATASET_ROOT,
    "train",
    "train_ground_truth.tsv",
)

SAMPLE_SIZE = 1000


# ============================================================
# NORMALIZATION
# ============================================================

def norm(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# LOAD CACHE
# ============================================================

print("Loading cached index...")

with open(
    INDEX_FILE,
    "rb",
) as f:

    cached = pickle.load(f)

rid_ids = cached["rid_ids"]

print(
    f"Candidate record codes: "
    f"{len(rid_ids):,}"
)

print(
    "Cached index loaded."
)


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

# Important:
# candidate file's s1_idx corresponds to the original
# S1 row order because the candidate generator enumerated
# s1_keys in S1 order.

s1_idx_to_id = dict(
    enumerate(
        s1["entity_id"].tolist()
    )
)

print(
    f"S1 rows: "
    f"{len(s1_idx_to_id):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

truth_by_s1 = {}

total_truth = 0

for row in gt.itertuples(
    index=False
):

    matches = str(
        row.matched_entity_ids
    ).strip()

    if not matches:
        continue

    for other in matches.split(","):

        other = other.strip()

        if not other:
            continue

        truth_by_s1.setdefault(
            row.source1_entity_id,
            set(),
        ).add(
            other
        )

        total_truth += 1


print(
    f"TRUE PAIRS: "
    f"{total_truth:,}"
)


# ============================================================
# STREAM PARQUET
# ============================================================

print(
    "\nScanning candidate parquet "
    "in batches..."
)

pf = pq.ParquetFile(
    CANDIDATE_FILE
)

print(
    f"Parquet row groups: "
    f"{pf.num_row_groups:,}"
)

# We only need to know whether a true pair
# was recovered.
#
# So we do NOT store all 527M candidate pairs.
#
# found_by_s1 stores only recovered TRUE pairs.

found_by_s1 = {}

rows_scanned = 0

for batch_no, batch in enumerate(
    pf.iter_batches(
        batch_size=2_000_000,
        columns=[
            "s1_idx",
            "cand_code",
        ],
    )
):

    df = batch.to_pandas()

    for row in df.itertuples(
        index=False
    ):

        s1_idx = int(
            row.s1_idx
        )

        cand_code = int(
            row.cand_code
        )

        s1_id = s1_idx_to_id.get(
            s1_idx
        )

        if s1_id is None:
            continue

        true_ids = truth_by_s1.get(
            s1_id
        )

        if not true_ids:
            continue

        if (
            cand_code < 0
            or cand_code >= len(rid_ids)
        ):
            continue

        other_id = rid_ids[
            cand_code
        ]

        if other_id in true_ids:

            found_by_s1.setdefault(
                s1_id,
                set(),
            ).add(
                other_id
            )

    rows_scanned += len(df)

    print(
        f"  scanned "
        f"{rows_scanned:,} / "
        f"{pf.metadata.num_rows:,} "
        f"candidate rows",
        flush=True,
    )


# ============================================================
# FIND MISSED PAIRS
# ============================================================

missed = []

recovered = 0

for s1_id, true_ids in (
    truth_by_s1.items()
):

    found = found_by_s1.get(
        s1_id,
        set(),
    )

    recovered += len(
        found
    )

    for other_id in (
        true_ids - found
    ):

        missed.append(
            (
                s1_id,
                other_id,
            )
        )


print()
print(
    "=" * 80
)

print(
    f"TRUE PAIRS: "
    f"{total_truth:,}"
)

print(
    f"RECOVERED: "
    f"{recovered:,}"
)

print(
    f"MISSED: "
    f"{len(missed):,}"
)

print(
    f"RECALL: "
    f"{recovered / total_truth:.4f}"
)

print(
    "=" * 80
)


if not missed:

    print(
        "No missed pairs!"
    )

    raise SystemExit


# ============================================================
# SAMPLE MISSES
# ============================================================

random.seed(42)

sample = (
    random.sample(
        missed,
        min(
            SAMPLE_SIZE,
            len(missed),
        ),
    )
)


# ============================================================
# LOAD REQUIRED RECORDS
# ============================================================

missed_s1_ids = {
    x[0]
    for x in sample
}

missed_other_ids = {
    x[1]
    for x in sample
}


print(
    "\nLoading relevant S1 records..."
)

s1_small = s1[
    s1["entity_id"].isin(
        missed_s1_ids
    )
]

s1_map = {
    r.entity_id: r
    for r in s1_small.itertuples(
        index=False
    )
}


def load_other(path):

    print(
        f"Loading "
        f"{os.path.basename(path)}..."
    )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    df = df[
        df["entity_id"].isin(
            missed_other_ids
        )
    ]

    return {
        r.entity_id: r
        for r in df.itertuples(
            index=False
        )
    }


s2_map = load_other(
    S2_FILE
)

s3_map = load_other(
    S3_FILE
)

other_map = {}

other_map.update(
    s2_map
)

other_map.update(
    s3_map
)


# ============================================================
# CLASSIFICATION
# ============================================================

stats = Counter()

examples = {}


def classify(
    s1row,
    orow,
):

    name1 = norm(
        s1row.business_name
    )

    name2 = norm(
        orow.business_name
    )

    addr1 = norm(
        s1row.business_address
    )

    addr2 = norm(
        orow.business_address
    )

    if not name1 or not name2:

        category = "empty_name"

    else:

        name_ratio = ratio(
            name1,
            name2,
        )

        addr_ratio = ratio(
            addr1,
            addr2,
        )

        name_token = token_set_ratio(
            name1,
            name2,
        )

        addr_token = token_set_ratio(
            addr1,
            addr2,
        )

        name_close = (
            name_ratio >= 85
            or name_token >= 90
        )

        addr_close = (
            addr_ratio >= 85
            or addr_token >= 90
        )

        if (
            name_ratio == 100
            and name1 == name2
        ):

            category = "exact_name"

        elif (
            name_close
            and addr_close
        ):

            category = "both_close"

        elif name_close:

            category = "name_close"

        elif addr_close:

            category = "address_close"

        else:

            category = "both_far"

    # Non-Latin detection
    raw1 = str(
        s1row.business_name
    )

    raw2 = str(
        orow.business_name
    )

    if any(
        ord(c) > 127
        for c in raw1 + raw2
    ):

        if category not in {
            "exact_name",
            "both_close",
        }:

            category = "nonlatin"

    return category


# ============================================================
# DIAGNOSE
# ============================================================

for s1_id, other_id in sample:

    s1row = s1_map.get(
        s1_id
    )

    orow = other_map.get(
        other_id
    )

    if (
        s1row is None
        or orow is None
    ):
        continue

    category = classify(
        s1row,
        orow,
    )

    stats[category] += 1

    examples.setdefault(
        category,
        [],
    )

    if len(
        examples[category]
    ) < 5:

        examples[category].append(
            (
                s1_id,
                other_id,
                str(
                    s1row.business_name
                ),
                str(
                    orow.business_name
                ),
                str(
                    s1row.business_address
                ),
                str(
                    orow.business_address
                ),
            )
        )


# ============================================================
# RESULTS
# ============================================================

print()
print(
    "=" * 80
)

print(
    "MISSED-PAIR DIAGNOSIS"
)

print(
    "=" * 80
)

print(
    f"Sampled missed pairs: "
    f"{len(sample):,}"
)

print()

for category, count in (
    stats.most_common()
):

    pct = (
        count
        / len(sample)
        * 100
    )

    print(
        f"{category:20s}"
        f"{count:6d}"
        f" ({pct:5.1f}%)"
    )


# ============================================================
# EXAMPLES
# ============================================================

for category, rows in (
    examples.items()
):

    print()
    print(
        "=" * 80
    )

    print(
        f"EXAMPLES: "
        f"{category}"
    )

    print(
        "=" * 80
    )

    for row in rows:

        (
            s1_id,
            other_id,
            name1,
            name2,
            addr1,
            addr2,
        ) = row

        print()
        print(
            f"S1 ID:    {s1_id}"
        )

        print(
            f"OTHER ID: {other_id}"
        )

        print(
            f"NAME 1:   {name1}"
        )

        print(
            f"NAME 2:   {name2}"
        )

        print(
            f"ADDR 1:   {addr1}"
        )

        print(
            f"ADDR 2:   {addr2}"
        )


print()
print(
    "=" * 80
)

print(
    "DONE"
)

print(
    "=" * 80
)