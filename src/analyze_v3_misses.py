import os
import re
import pickle
import unicodedata
from collections import Counter, defaultdict

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
    "dataset"
)

CACHE_DIR = os.path.join(
    ROOT,
    "cache"
)

GT_PATH = os.path.join(
    DATASET_ROOT,
    "train",
    "train_ground_truth.tsv"
)

S1_PATH = os.path.join(
    DATASET_ROOT,
    "train",
    "train_source1.tsv"
)

S2_PATH = os.path.join(
    DATASET_ROOT,
    "train",
    "train_source2.tsv"
)

S3_PATH = os.path.join(
    DATASET_ROOT,
    "train",
    "train_source3.tsv"
)

CANDIDATE_PATH = os.path.join(
    CACHE_DIR,
    "train_v3_candidate_pairs.parquet"
)

ID_MAP_PATH = os.path.join(
    CACHE_DIR,
    "train_v3_id_maps.pkl"
)

OUTPUT_PATH = os.path.join(
    CACHE_DIR,
    "v3_missed_pairs_sample.tsv"
)


# ============================================================
# CONFIG
# ============================================================

# Number of missed pairs to save for detailed inspection.
SAMPLE_SIZE = 20_000

# Number of examples printed to terminal.
PRINT_EXAMPLES = 50


# ============================================================
# NORMALIZATION
# ============================================================

TOKEN_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)

NUM_RE = re.compile(
    r"\d+[a-zA-Z]?"
)


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
    )

    return text.strip()


def normalize_ascii(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text)
    )

    text = (
        text
        .encode(
            "ascii",
            "ignore"
        )
        .decode("ascii")
    )

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
    )

    return text.strip()


def tokens(text):

    if not text:
        return []

    return TOKEN_RE.findall(
        text
    )


def numbers(text):

    if not text:
        return []

    return NUM_RE.findall(
        text
    )


# ============================================================
# SIMILARITY FEATURES FOR DIAGNOSTICS
# ============================================================

def name_similarity(
    a,
    b
):

    if not a or not b:
        return 0.0

    return ratio(
        a,
        b
    )


def name_token_similarity(
    a,
    b
):

    if not a or not b:
        return 0.0

    return token_set_ratio(
        a,
        b
    )


def address_similarity(
    a,
    b
):

    if not a or not b:
        return 0.0

    return ratio(
        a,
        b
    )


def address_token_similarity(
    a,
    b
):

    if not a or not b:
        return 0.0

    return token_set_ratio(
        a,
        b
    )


def number_overlap(
    a,
    b
):

    na = set(
        numbers(a)
    )

    nb = set(
        numbers(b)
    )

    if not na or not nb:
        return 0.0

    return (
        len(na & nb)
        /
        max(
            len(na | nb),
            1
        )
    )


def has_non_latin(
    text
):

    if not text:
        return False

    for ch in str(text):

        if ch.isalpha():

            if ord(ch) > 127:
                return True

    return False


# ============================================================
# CLASSIFY MISSED PAIR
# ============================================================

def classify_pair(
    name1,
    name2,
    addr1,
    addr2
):

    nu1 = normalize_unicode(
        name1
    )

    nu2 = normalize_unicode(
        name2
    )

    na1 = normalize_ascii(
        name1
    )

    na2 = normalize_ascii(
        name2
    )

    au1 = normalize_unicode(
        addr1
    )

    au2 = normalize_unicode(
        addr2
    )

    name_r = name_similarity(
        nu1,
        nu2
    )

    name_token_r = name_token_similarity(
        nu1,
        nu2
    )

    ascii_name_r = name_similarity(
        na1,
        na2
    )

    addr_r = address_similarity(
        au1,
        au2
    )

    addr_token_r = address_token_similarity(
        au1,
        au2
    )

    num_r = number_overlap(
        au1,
        au2
    )

    nonlatin = (
        has_non_latin(name1)
        or
        has_non_latin(name2)
    )

    # --------------------------------------------------------
    # Classification
    #
    # These are diagnostic buckets, not model labels.
    # --------------------------------------------------------

    if nonlatin:

        category = (
            "cross_script"
        )

    elif (
        name_r >= 90
        and addr_r >= 90
    ):

        category = (
            "both_very_close"
        )

    elif (
        name_r >= 90
        and addr_r < 70
    ):

        category = (
            "name_strong_address_weak"
        )

    elif (
        name_r < 70
        and addr_r >= 90
    ):

        category = (
            "address_strong_name_weak"
        )

    elif (
        name_r >= 80
        and addr_r >= 80
    ):

        category = (
            "both_close"
        )

    elif name_r >= 80:

        category = (
            "name_close"
        )

    elif addr_r >= 80:

        category = (
            "address_close"
        )

    elif (
        ascii_name_r >= 80
        and name_r < 70
    ):

        category = (
            "transliteration_like"
        )

    else:

        category = (
            "both_far"
        )

    return {
        "name_ratio": name_r,
        "name_token_ratio": name_token_r,
        "ascii_name_ratio": ascii_name_r,
        "address_ratio": addr_r,
        "address_token_ratio": addr_token_r,
        "number_overlap": num_r,
        "cross_script": int(nonlatin),
        "category": category,
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_sources():

    print("\nLoading Source 1...")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        keep_default_na=False
    )

    print(
        f"S1: {len(s1):,}"
    )

    print("\nLoading Source 2...")

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        keep_default_na=False
    )

    print(
        f"S2: {len(s2):,}"
    )

    print("\nLoading Source 3...")

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        keep_default_na=False
    )

    print(
        f"S3: {len(s3):,}"
    )

    return s1, s2, s3


# ============================================================
# BUILD TARGET LOOKUP
# ============================================================

def build_target_lookup(
    s2,
    s3
):

    print(
        "\nBuilding S2/S3 entity lookup..."
    )

    lookup = {}

    for row in s2.itertuples(
        index=False
    ):

        lookup[
            row.entity_id
        ] = (
            row.business_name,
            row.business_address,
            row.country,
            "source2"
        )

    for row in s3.itertuples(
        index=False
    ):

        lookup[
            row.entity_id
        ] = (
            row.business_name,
            row.business_address,
            row.country,
            "source3"
        )

    print(
        f"Target records: "
        f"{len(lookup):,}"
    )

    return lookup


# ============================================================
# LOAD CANDIDATE PAIRS INTO SET
#
# We need membership checks for the GT pairs.
# ============================================================

def load_candidate_pairs():

    print(
        "\nLoading candidate pairs..."
    )

    pf = pq.ParquetFile(
        CANDIDATE_PATH
    )

    candidate_set = set()

    total = 0

    batch_no = 0

    for batch in pf.iter_batches(
        batch_size=5_000_000
    ):

        batch_no += 1

        s1_arr = batch[
            "s1_idx"
        ].to_numpy()

        cand_arr = batch[
            "cand_code"
        ].to_numpy()

        for s1_idx, cand_code in zip(
            s1_arr,
            cand_arr
        ):

            candidate_set.add(
                (
                    int(s1_idx),
                    int(cand_code)
                )
            )

        total += len(
            s1_arr
        )

        print(
            f"  batch={batch_no:,} "
            f"rows={total:,}",
            flush=True
        )

    print(
        f"\nUnique candidate pairs: "
        f"{len(candidate_set):,}"
    )

    return candidate_set


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "V3 MISSED-PAIR DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required = [
        GT_PATH,
        S1_PATH,
        S2_PATH,
        S3_PATH,
        CANDIDATE_PATH,
        ID_MAP_PATH,
    ]

    print(
        "\nChecking files..."
    )

    for path in required:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"\nMissing file:\n{path}"
            )

        print(
            f"✓ {path}"
        )

    # --------------------------------------------------------
    # Load ID maps
    # --------------------------------------------------------

    print(
        "\nLoading V3 ID maps..."
    )

    with open(
        ID_MAP_PATH,
        "rb"
    ) as f:

        maps = pickle.load(
            f
        )

    s1_ids = maps[
        "s1_ids"
    ]

    rid_ids = maps[
        "rid_ids"
    ]

    s1_to_idx = {
        eid: i
        for i, eid in enumerate(
            s1_ids
        )
    }

    rid_to_code = {
        eid: i
        for i, eid in enumerate(
            rid_ids
        )
    }

    print(
        f"S1 IDs: {len(s1_ids):,}"
    )

    print(
        f"Indexed target IDs: "
        f"{len(rid_ids):,}"
    )

    # --------------------------------------------------------
    # Load sources
    # --------------------------------------------------------

    s1, s2, s3 = load_sources()

    # --------------------------------------------------------
    # Build target lookup
    # --------------------------------------------------------

    target_lookup = build_target_lookup(
        s2,
        s3
    )

    # --------------------------------------------------------
    # Load candidate set
    # --------------------------------------------------------

    candidate_set = load_candidate_pairs()

    # --------------------------------------------------------
    # Build S1 lookup
    # --------------------------------------------------------

    print(
        "\nBuilding S1 lookup..."
    )

    s1_lookup = {}

    for row in s1.itertuples(
        index=False
    ):

        s1_lookup[
            row.entity_id
        ] = (
            row.business_name,
            row.business_address,
            row.country
        )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # --------------------------------------------------------
    # Diagnostic counters
    # --------------------------------------------------------

    total_truth = 0

    missed_total = 0

    missed_not_indexed = 0

    missed_indexed = 0

    category_counter = Counter()

    # Similarity buckets
    name_buckets = Counter()
    address_buckets = Counter()

    cross_script_count = 0

    # --------------------------------------------------------
    # Save detailed samples
    # --------------------------------------------------------

    sample_rows = []

    # --------------------------------------------------------
    # Process GT
    # --------------------------------------------------------

    print(
        "\nAnalyzing missed pairs..."
    )

    for row_no, row in enumerate(
        gt.itertuples(
            index=False
        )
    ):

        s1_id = row.source1_entity_id

        s1_idx = s1_to_idx.get(
            s1_id
        )

        if s1_idx is None:
            continue

        value = str(
            row.matched_entity_ids
        ).strip()

        if not value:
            continue

        targets = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        for target_id in targets:

            total_truth += 1

            target_code = (
                rid_to_code.get(
                    target_id
                )
            )

            # ------------------------------------------------
            # If target is not in index, blocking definitely
            # missed it.
            # ------------------------------------------------

            if target_code is None:

                missed_total += 1

                missed_not_indexed += 1

                if (
                    len(sample_rows)
                    <
                    SAMPLE_SIZE
                ):

                    s1_data = (
                        s1_lookup.get(
                            s1_id,
                            ("", "", "")
                        )
                    )

                    target_data = (
                        target_lookup.get(
                            target_id,
                            ("", "", "")
                        )
                    )

                    name1, addr1, country1 = (
                        s1_data
                    )

                    name2, addr2, country2, source = (
                        target_data
                    )

                    sim = classify_pair(
                        name1,
                        name2,
                        addr1,
                        addr2
                    )

                    sample_rows.append(
                        {
                            "s1_entity_id": s1_id,
                            "target_entity_id": target_id,
                            "target_source": source,
                            "country_s1": country1,
                            "country_target": country2,
                            "name_s1": name1,
                            "name_target": name2,
                            "address_s1": addr1,
                            "address_target": addr2,
                            "indexed": 0,
                            "candidate": 0,
                            **sim,
                        }
                    )

                continue

            # ------------------------------------------------
            # Check whether actual pair exists in candidate set
            # ------------------------------------------------

            pair = (
                int(s1_idx),
                int(target_code)
            )

            if pair in candidate_set:

                # Correctly blocked.
                continue

            # ------------------------------------------------
            # Missed even though target was indexed.
            # ------------------------------------------------

            missed_total += 1

            missed_indexed += 1

            # ------------------------------------------------
            # Get actual records
            # ------------------------------------------------

            s1_data = (
                s1_lookup.get(
                    s1_id,
                    ("", "", "")
                )
            )

            target_data = (
                target_lookup.get(
                    target_id,
                    ("", "", "")
                )
            )

            name1, addr1, country1 = (
                s1_data
            )

            name2, addr2, country2, source = (
                target_data
            )

            # ------------------------------------------------
            # Similarity analysis
            # ------------------------------------------------

            sim = classify_pair(
                name1,
                name2,
                addr1,
                addr2
            )

            category = sim[
                "category"
            ]

            category_counter[
                category
            ] += 1

            # ------------------------------------------------
            # Name buckets
            # ------------------------------------------------

            nr = sim[
                "name_ratio"
            ]

            if nr >= 95:
                name_buckets[
                    "95-100"
                ] += 1

            elif nr >= 90:
                name_buckets[
                    "90-95"
                ] += 1

            elif nr >= 80:
                name_buckets[
                    "80-90"
                ] += 1

            elif nr >= 70:
                name_buckets[
                    "70-80"
                ] += 1

            elif nr >= 50:
                name_buckets[
                    "50-70"
                ] += 1

            else:
                name_buckets[
                    "<50"
                ] += 1

            # ------------------------------------------------
            # Address buckets
            # ------------------------------------------------

            ar = sim[
                "address_ratio"
            ]

            if ar >= 95:
                address_buckets[
                    "95-100"
                ] += 1

            elif ar >= 90:
                address_buckets[
                    "90-95"
                ] += 1

            elif ar >= 80:
                address_buckets[
                    "80-90"
                ] += 1

            elif ar >= 70:
                address_buckets[
                    "70-80"
                ] += 1

            elif ar >= 50:
                address_buckets[
                    "50-70"
                ] += 1

            else:
                address_buckets[
                    "<50"
                ] += 1

            if sim[
                "cross_script"
            ]:

                cross_script_count += 1

            # ------------------------------------------------
            # Save sample
            # ------------------------------------------------

            if (
                len(sample_rows)
                <
                SAMPLE_SIZE
            ):

                sample_rows.append(
                    {
                        "s1_entity_id": s1_id,
                        "target_entity_id": target_id,
                        "target_source": source,
                        "country_s1": country1,
                        "country_target": country2,
                        "name_s1": name1,
                        "name_target": name2,
                        "address_s1": addr1,
                        "address_target": addr2,
                        "indexed": 1,
                        "candidate": 0,
                        **sim,
                    }
                )

        if (
            row_no + 1
        ) % 100_000 == 0:

            print(
                f"  GT rows "
                f"{row_no + 1:,}/"
                f"{len(gt):,} "
                f"truth={total_truth:,} "
                f"missed={missed_total:,}",
                flush=True
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "V3 MISSED-PAIR DIAGNOSTIC RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"TOTAL TRUE PAIRS       : "
        f"{total_truth:,}"
    )

    print(
        f"TOTAL MISSED           : "
        f"{missed_total:,}"
    )

    print(
        f"MISSED - NOT INDEXED  : "
        f"{missed_not_indexed:,}"
    )

    print(
        f"MISSED - INDEXED       : "
        f"{missed_indexed:,}"
    )

    if total_truth:

        print(
            f"MISS RATE               : "
            f"{missed_total / total_truth:.4%}"
        )

        print(
            f"NOT INDEXED RATE       : "
            f"{missed_not_indexed / total_truth:.4%}"
        )

        print(
            f"INDEXED MISS RATE      : "
            f"{missed_indexed / total_truth:.4%}"
        )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        "MISSED PAIR CATEGORIES"
    )

    print(
        "-" * 70
    )

    for category, count in (
        category_counter
        .most_common()
    ):

        pct = (
            count
            /
            max(
                missed_indexed,
                1
            )
            *
            100
        )

        print(
            f"{category:<32}"
            f"{count:>10,}"
            f"  {pct:>7.2f}%"
        )

    # ========================================================
    # CROSS SCRIPT
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        "CROSS-SCRIPT"
    )

    print(
        "-" * 70
    )

    print(
        f"Cross-script indexed misses: "
        f"{cross_script_count:,}"
    )

    if missed_indexed:

        print(
            f"Percentage: "
            f"{cross_script_count / missed_indexed:.2%}"
        )

    # ========================================================
    # NAME DISTRIBUTION
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        "NAME SIMILARITY DISTRIBUTION"
    )

    print(
        "-" * 70
    )

    name_order = [
        "95-100",
        "90-95",
        "80-90",
        "70-80",
        "50-70",
        "<50",
    ]

    for bucket in name_order:

        count = name_buckets[
            bucket
        ]

        pct = (
            count
            /
            max(
                missed_indexed,
                1
            )
            *
            100
        )

        print(
            f"{bucket:<10}"
            f"{count:>10,}"
            f"  {pct:>7.2f}%"
        )

    # ========================================================
    # ADDRESS DISTRIBUTION
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        "ADDRESS SIMILARITY DISTRIBUTION"
    )

    print(
        "-" * 70
    )

    address_order = [
        "95-100",
        "90-95",
        "80-90",
        "70-80",
        "50-70",
        "<50",
    ]

    for bucket in address_order:

        count = address_buckets[
            bucket
        ]

        pct = (
            count
            /
            max(
                missed_indexed,
                1
            )
            *
            100
        )

        print(
            f"{bucket:<10}"
            f"{count:>10,}"
            f"  {pct:>7.2f}%"
        )

    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    if sample_rows:

        sample_df = pd.DataFrame(
            sample_rows
        )

        sample_df.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False
        )

        print()
        print(
            f"Saved detailed sample:"
        )

        print(
            OUTPUT_PATH
        )

        print(
            f"Sample rows: "
            f"{len(sample_df):,}"
        )

    # ========================================================
    # PRINT EXAMPLES
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        f"FIRST {PRINT_EXAMPLES} MISSED EXAMPLES"
    )

    print(
        "=" * 70
    )

    for i, row in enumerate(
        sample_rows[
            :PRINT_EXAMPLES
        ],
        start=1
    ):

        print()
        print(
            f"[{i}] "
            f"{row['category']}"
        )

        print(
            f"S1:     "
            f"{row['name_s1']}"
        )

        print(
            f"TARGET: "
            f"{row['name_target']}"
        )

        print(
            f"S1 ADDR: "
            f"{row['address_s1']}"
        )

        print(
            f"T ADDR:  "
            f"{row['address_target']}"
        )

        print(
            f"name={row['name_ratio']:.1f} "
            f"addr={row['address_ratio']:.1f} "
            f"token_name={row['name_token_ratio']:.1f} "
            f"token_addr={row['address_token_ratio']:.1f}"
        )

        print(
            f"source={row['target_source']} "
            f"country={row['country_s1']}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()