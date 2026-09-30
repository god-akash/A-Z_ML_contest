import os
import pickle
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

# We do NOT load all 462M candidates into RAM.
#
# We only keep the TRUE ground-truth pairs in RAM.
#
# Maximum detailed examples saved.
SAMPLE_SIZE = 20_000


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


def numbers(text):

    if not text:
        return []

    return NUM_RE.findall(
        text
    )


def has_non_latin(text):

    if not text:
        return False

    for ch in str(text):

        if ch.isalpha() and ord(ch) > 127:
            return True

    return False


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(
    name1,
    name2,
    addr1,
    addr2
):

    n1 = normalize_unicode(
        name1
    )

    n2 = normalize_unicode(
        name2
    )

    a1 = normalize_unicode(
        addr1
    )

    a2 = normalize_unicode(
        addr2
    )

    ascii_n1 = normalize_ascii(
        name1
    )

    ascii_n2 = normalize_ascii(
        name2
    )

    if n1 and n2:

        name_ratio = ratio(
            n1,
            n2
        )

        name_token_ratio = token_set_ratio(
            n1,
            n2
        )

    else:

        name_ratio = 0.0
        name_token_ratio = 0.0

    if a1 and a2:

        address_ratio = ratio(
            a1,
            a2
        )

        address_token_ratio = token_set_ratio(
            a1,
            a2
        )

    else:

        address_ratio = 0.0
        address_token_ratio = 0.0

    if ascii_n1 and ascii_n2:

        ascii_name_ratio = ratio(
            ascii_n1,
            ascii_n2
        )

    else:

        ascii_name_ratio = 0.0

    nums1 = set(
        numbers(a1)
    )

    nums2 = set(
        numbers(a2)
    )

    if nums1 and nums2:

        number_overlap = (
            len(nums1 & nums2)
            /
            len(nums1 | nums2)
        )

    else:

        number_overlap = 0.0

    cross_script = (
        has_non_latin(name1)
        or
        has_non_latin(name2)
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if cross_script:

        category = "cross_script"

    elif (
        name_ratio >= 90
        and address_ratio >= 90
    ):

        category = "both_very_close"

    elif (
        name_ratio >= 90
        and address_ratio < 70
    ):

        category = "name_strong_address_weak"

    elif (
        name_ratio < 70
        and address_ratio >= 90
    ):

        category = "address_strong_name_weak"

    elif (
        name_ratio >= 80
        and address_ratio >= 80
    ):

        category = "both_close"

    elif name_ratio >= 80:

        category = "name_close"

    elif address_ratio >= 80:

        category = "address_close"

    elif (
        ascii_name_ratio >= 80
        and name_ratio < 70
    ):

        category = "transliteration_like"

    else:

        category = "both_far"

    return {
        "name_ratio": name_ratio,
        "name_token_ratio": name_token_ratio,
        "ascii_name_ratio": ascii_name_ratio,
        "address_ratio": address_ratio,
        "address_token_ratio": address_token_ratio,
        "number_overlap": number_overlap,
        "cross_script": int(cross_script),
        "category": category,
    }


# ============================================================
# LOAD SOURCES
# ============================================================

def load_sources():

    print("\nLoading S1...")

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

    print("\nLoading S2...")

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

    print("\nLoading S3...")

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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEMORY-SAFE V3 MISSED-PAIR DIAGNOSTIC")
    print("=" * 70)

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

    print("\nChecking files...")

    for path in required:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nMissing file:\n{path}"
            )

        print(
            f"✓ {path}"
        )

    # --------------------------------------------------------
    # Load cached ID maps
    # --------------------------------------------------------

    print(
        "\nLoading cached V3 ID maps..."
    )

    with open(
        ID_MAP_PATH,
        "rb"
    ) as f:

        maps = pickle.load(f)

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
        f"S1 records: "
        f"{len(s1_ids):,}"
    )

    print(
        f"Indexed target records: "
        f"{len(rid_ids):,}"
    )

    # --------------------------------------------------------
    # Load S1/S2/S3
    # --------------------------------------------------------

    s1, s2, s3 = load_sources()

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
    # Build target lookup
    # --------------------------------------------------------

    print(
        "\nBuilding target lookup..."
    )

    target_lookup = {}

    for row in s2.itertuples(
        index=False
    ):

        target_lookup[
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

        target_lookup[
            row.entity_id
        ] = (
            row.business_name,
            row.business_address,
            row.country,
            "source3"
        )

    print(
        f"Target lookup: "
        f"{len(target_lookup):,}"
    )

    # ========================================================
    # STEP 1
    #
    # Build ONLY the TRUE PAIR set.
    #
    # 7.6M pairs is manageable.
    #
    # DO NOT load candidate pairs into RAM.
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1: BUILD TRUE PAIR SET")
    print("=" * 70)

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    true_pairs = set()

    total_truth = 0

    not_indexed = set()

    for row_no, row in enumerate(
        gt.itertuples(index=False)
    ):

        s1_idx = s1_to_idx.get(
            row.source1_entity_id
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

        total_truth += len(
            targets
        )

        for target_id in targets:

            code = rid_to_code.get(
                target_id
            )

            if code is None:

                # This target never entered the V3 index.
                not_indexed.add(
                    (
                        s1_idx,
                        target_id
                    )
                )

                continue

            packed = (
                s1_idx
                *
                (
                    len(rid_ids) + 1
                )
                +
                code
            )

            true_pairs.add(
                packed
            )

        if (
            row_no + 1
        ) % 500_000 == 0:

            print(
                f"  GT rows: "
                f"{row_no + 1:,}/"
                f"{len(gt):,} "
                f"truth={total_truth:,}",
                flush=True
            )

    print()
    print(
        f"FULL TRUE PAIRS: "
        f"{total_truth:,}"
    )

    print(
        f"TRUE PAIRS IN INDEX: "
        f"{len(true_pairs):,}"
    )

    print(
        f"TRUE TARGETS NOT INDEXED: "
        f"{len(not_indexed):,}"
    )

    # ========================================================
    # STEP 2
    #
    # Stream 462M candidates.
    #
    # We DO NOT store them.
    #
    # We simply remove recovered true pairs from true_pairs.
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2: STREAM CANDIDATES")
    print("=" * 70)

    print(
        "IMPORTANT: "
        "Candidate rows are NOT stored in RAM."
    )

    pf = pq.ParquetFile(
        CANDIDATE_PATH
    )

    processed = 0

    recovered = 0

    batch_no = 0

    for batch in pf.iter_batches(
        batch_size=2_000_000
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

            packed = (
                int(s1_idx)
                *
                (
                    len(rid_ids) + 1
                )
                +
                int(cand_code)
            )

            if packed in true_pairs:

                true_pairs.remove(
                    packed
                )

                recovered += 1

        processed += len(
            s1_arr
        )

        if batch_no % 10 == 0:

            print(
                f"  processed="
                f"{processed:,} "
                f"recovered="
                f"{recovered:,} "
                f"remaining_true="
                f"{len(true_pairs):,}",
                flush=True
            )

    # ========================================================
    # STEP 3
    #
    # true_pairs now contains ONLY indexed true pairs
    # that V3 failed to generate.
    # ========================================================

    indexed_misses = true_pairs

    print()
    print("=" * 70)
    print("STEP 3: MISS SUMMARY")
    print("=" * 70)

    print(
        f"FULL TRUE PAIRS     : "
        f"{total_truth:,}"
    )

    print(
        f"RECOVERED           : "
        f"{recovered:,}"
    )

    print(
        f"NOT INDEXED         : "
        f"{len(not_indexed):,}"
    )

    print(
        f"INDEXED BUT MISSED  : "
        f"{len(indexed_misses):,}"
    )

    print(
        f"TOTAL MISSED        : "
        f"{len(not_indexed) + len(indexed_misses):,}"
    )

    # ========================================================
    # STEP 4
    #
    # Analyze indexed misses.
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4: ANALYZE INDEXED MISSES")
    print("=" * 70)

    category_counter = Counter()

    name_buckets = Counter()

    address_buckets = Counter()

    cross_script_count = 0

    sample_rows = []

    # --------------------------------------------------------
    # Convert packed indexed misses back to IDs
    # --------------------------------------------------------

    divisor = (
        len(rid_ids) + 1
    )

    # We only analyze up to SAMPLE_SIZE detailed rows.
    #
    # Statistics are still calculated for all indexed misses.
    # --------------------------------------------------------

    for miss_no, packed in enumerate(
        indexed_misses
    ):

        s1_idx = (
            packed // divisor
        )

        target_code = (
            packed % divisor
        )

        if (
            s1_idx >= len(s1_ids)
            or
            target_code >= len(rid_ids)
        ):

            continue

        s1_id = s1_ids[
            s1_idx
        ]

        target_id = rid_ids[
            target_code
        ]

        s1_data = s1_lookup.get(
            s1_id,
            ("", "", "")
        )

        target_data = target_lookup.get(
            target_id,
            ("", "", "", "")
        )

        name1, addr1, country1 = (
            s1_data
        )

        name2, addr2, country2, source = (
            target_data
        )

        sim = calculate_similarity(
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

        # ----------------------------------------------------
        # Name bucket
        # ----------------------------------------------------

        nr = sim[
            "name_ratio"
        ]

        if nr >= 95:
            name_buckets["95-100"] += 1

        elif nr >= 90:
            name_buckets["90-95"] += 1

        elif nr >= 80:
            name_buckets["80-90"] += 1

        elif nr >= 70:
            name_buckets["70-80"] += 1

        elif nr >= 50:
            name_buckets["50-70"] += 1

        else:
            name_buckets["<50"] += 1

        # ----------------------------------------------------
        # Address bucket
        # ----------------------------------------------------

        ar = sim[
            "address_ratio"
        ]

        if ar >= 95:
            address_buckets["95-100"] += 1

        elif ar >= 90:
            address_buckets["90-95"] += 1

        elif ar >= 80:
            address_buckets["80-90"] += 1

        elif ar >= 70:
            address_buckets["70-80"] += 1

        elif ar >= 50:
            address_buckets["50-70"] += 1

        else:
            address_buckets["<50"] += 1

        if sim[
            "cross_script"
        ]:

            cross_script_count += 1

        # ----------------------------------------------------
        # Save sample
        # ----------------------------------------------------

        if len(sample_rows) < SAMPLE_SIZE:

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
                    **sim
                }
            )

    # ========================================================
    # NOT INDEXED SAMPLE
    # ========================================================

    print(
        "\nAnalyzing not-indexed examples..."
    )

    for s1_idx, target_id in list(
        not_indexed
    )[
        :SAMPLE_SIZE
    ]:

        if (
            s1_idx >= len(s1_ids)
        ):

            continue

        s1_id = s1_ids[
            s1_idx
        ]

        s1_data = s1_lookup.get(
            s1_id,
            ("", "", "")
        )

        target_data = target_lookup.get(
            target_id,
            ("", "", "", "")
        )

        name1, addr1, country1 = (
            s1_data
        )

        name2, addr2, country2, source = (
            target_data
        )

        sim = calculate_similarity(
            name1,
            name2,
            addr1,
            addr2
        )

        if len(sample_rows) < SAMPLE_SIZE:

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
                    **sim
                }
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MISS DIAGNOSTIC")
    print("=" * 70)

    print(
        f"TRUE PAIRS          : "
        f"{total_truth:,}"
    )

    print(
        f"RECOVERED           : "
        f"{recovered:,}"
    )

    print(
        f"INDEXED MISSES      : "
        f"{len(indexed_misses):,}"
    )

    print(
        f"NOT INDEXED         : "
        f"{len(not_indexed):,}"
    )

    print(
        f"TOTAL MISSES        : "
        f"{len(indexed_misses) + len(not_indexed):,}"
    )

    # ========================================================
    # CATEGORIES
    # ========================================================

    print()
    print("-" * 70)
    print("INDEXED MISS CATEGORIES")
    print("-" * 70)

    for category, count in (
        category_counter.most_common()
    ):

        pct = (
            count
            /
            max(
                len(indexed_misses),
                1
            )
            *
            100
        )

        print(
            f"{category:<35}"
            f"{count:>10,}"
            f"  {pct:>7.2f}%"
        )

    # ========================================================
    # CROSS SCRIPT
    # ========================================================

    print()
    print("-" * 70)
    print("CROSS-SCRIPT")
    print("-" * 70)

    print(
        f"Cross-script indexed misses: "
        f"{cross_script_count:,}"
    )

    if indexed_misses:

        print(
            f"Percentage: "
            f"{cross_script_count / len(indexed_misses):.2%}"
        )

    # ========================================================
    # NAME DISTRIBUTION
    # ========================================================

    print()
    print("-" * 70)
    print("NAME SIMILARITY")
    print("-" * 70)

    for bucket in [
        "95-100",
        "90-95",
        "80-90",
        "70-80",
        "50-70",
        "<50",
    ]:

        count = name_buckets[
            bucket
        ]

        pct = (
            count
            /
            max(
                len(indexed_misses),
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
    print("-" * 70)
    print("ADDRESS SIMILARITY")
    print("-" * 70)

    for bucket in [
        "95-100",
        "90-95",
        "80-90",
        "70-80",
        "50-70",
        "<50",
    ]:

        count = address_buckets[
            bucket
        ]

        pct = (
            count
            /
            max(
                len(indexed_misses),
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
            f"Saved sample:"
        )

        print(
            OUTPUT_PATH
        )

        print(
            f"Rows saved: "
            f"{len(sample_df):,}"
        )

    # ========================================================
    # PRINT EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("FIRST MISSED EXAMPLES")
    print("=" * 70)

    for i, row in enumerate(
        sample_rows[:30],
        start=1
    ):

        print()
        print(
            f"[{i}] "
            f"{row['category']}"
        )

        print(
            f"S1 NAME : "
            f"{row['name_s1']}"
        )

        print(
            f"T NAME  : "
            f"{row['name_target']}"
        )

        print(
            f"S1 ADDR : "
            f"{row['address_s1']}"
        )

        print(
            f"T ADDR  : "
            f"{row['address_target']}"
        )

        print(
            f"name={row['name_ratio']:.1f} "
            f"addr={row['address_ratio']:.1f} "
            f"name_token={row['name_token_ratio']:.1f} "
            f"addr_token={row['address_token_ratio']:.1f}"
        )

        print(
            f"indexed={row['indexed']} "
            f"source={row['target_source']}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()