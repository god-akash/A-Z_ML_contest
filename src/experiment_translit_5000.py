# src/experiment_translit_5000.py
#
# Purpose:
#   Test whether transliterated-name n-gram blocking adds useful recall
#   before modifying the full 2.2M-row blocker.
#
# Experiment:
#   5,000 S1 records
#   Compare:
#       BASELINE signals
#       BASELINE + transliterated n-gram signal
#
# This does NOT modify the existing train candidate cache.

import os
import re
import unicodedata
import zlib
import random
import pickle

from collections import defaultdict
from itertools import combinations

import pandas as pd
from rapidfuzz.fuzz import token_set_ratio
from unidecode import unidecode


# ============================================================
# CONFIG
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

TRAIN_ROOT = os.path.join(
    DATASET_ROOT,
    "train"
)

SAMPLE_SIZE = 5000

KEY_CAP = 300

NGRAM_K = 4
NGRAM_N = 4

RANDOM_SEED = 42


# ============================================================
# ORIGINAL SIGNALS
# ============================================================

BASE_SIGNALS = [
    "nx",
    "ns",
    "nc",
    "ap",
    "an",
    "ua",
    "aa",
    "ng",
]

ALL_SIGNALS = BASE_SIGNALS + [
    "nt"
]

BASE_SIG_ID = {
    s: i
    for i, s in enumerate(BASE_SIGNALS)
}

ALL_SIG_ID = {
    s: i
    for i, s in enumerate(ALL_SIGNALS)
}


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "pvt",
    "ltd",
    "private",
    "limited",
    "llc",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "llp",
    "plc",
    "lp",
}


ADDRESS_GENERIC = frozenset({
    "road",
    "street",
    "avenue",
    "drive",
    "lane",
    "floor",
    "building",
    "block",
    "city",
    "district",
    "state",
    "country",
    "area",
    "plot",
    "unit",
    "shop",
})


TLD_RE = re.compile(
    r"\.(com|net|org|info|biz|co\.in|in|us|fr)\b",
    flags=re.IGNORECASE
)

TOKEN_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)

NUM_RE = re.compile(
    r"\d+[a-zA-Z]?"
)


# ============================================================
# HASH
# ============================================================

def crc(key: str) -> int:
    return zlib.crc32(
        key.encode("utf-8")
    )


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def unicode_normalize(text) -> str:

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

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def ascii_normalize(text) -> str:

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

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text.casefold()
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def transliterate_normalize(text) -> str:

    if not text:
        return ""

    s = str(text)

    # IMPORTANT:
    # Do not transliterate ASCII names.
    # Existing signals already handle them.
    if s.isascii():
        return ""

    text = unidecode(
        s
    ).casefold()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def tokenize(text):

    if not text:
        return []

    return TOKEN_RE.findall(
        text
    )


def normalize_name(text):

    text = unicode_normalize(
        text
    )

    if not text:
        return ""

    text = TLD_RE.sub(
        "",
        text
    )

    return " ".join(
        t
        for t in tokenize(text)
        if t not in LEGAL_SUFFIXES
    )


def normalize_address(text):

    return unicode_normalize(
        text
    )


def extract_numbers(text):

    if not text:
        return []

    return NUM_RE.findall(
        text
    )


def address_tokens(text):

    out = []

    for t in tokenize(text):

        if any(
            c.isdigit()
            for c in t
        ):
            out.append(t)

        elif len(t) >= 4:
            out.append(t)

    return out


# ============================================================
# N-GRAMS
# ============================================================

def name_ngrams(
    name_u,
    k=NGRAM_K
):

    s = name_u.replace(
        " ",
        ""
    )

    if len(s) < k:

        return [
            s
        ] if s else []

    return [
        s[i:i + k]
        for i in range(
            len(s) - k + 1
        )
    ]


# ============================================================
# BUILD KEYS
# ============================================================

def row_keys(
    country,
    name_u,
    addr_u,
    include_translit=False
):

    out = []

    ntoks = tokenize(
        name_u
    )


    # ========================================================
    # 1. EXACT NORMALIZED NAME
    # ========================================================

    if ntoks:

        out.append(
            (
                "nx",
                crc(
                    f"nx|{country}|"
                    f"{' '.join(ntoks)}"
                )
            )
        )


        # ====================================================
        # 2. SORTED TOKENS
        # ====================================================

        out.append(
            (
                "ns",
                crc(
                    f"ns|{country}|"
                    f"{' '.join(sorted(set(ntoks)))}"
                )
            )
        )


    # ========================================================
    # 3. COMPACT NAME
    # ========================================================

    compact = "".join(
        ntoks
    )

    if len(compact) >= 6:

        out.append(
            (
                "nc",
                crc(
                    f"nc|{country}|"
                    f"{compact}"
                )
            )
        )


    # ========================================================
    # 4. UNICODE FIRST/LAST
    # ========================================================

    if ntoks:

        first = ntoks[0]
        last = ntoks[-1]

        if len(first) >= 2:

            out.append(
                (
                    "ua",
                    crc(
                        f"ua|{country}|"
                        f"{first}|{last}"
                    )
                )
            )


    # ========================================================
    # 5. ASCII FIRST/LAST
    # ========================================================

    ascii_name = ascii_normalize(
        name_u
    )

    atoks = tokenize(
        ascii_name
    )

    if atoks:

        first = atoks[0]
        last = atoks[-1]

        if len(first) >= 2:

            out.append(
                (
                    "aa",
                    crc(
                        f"aa|{country}|"
                        f"{first}|{last}"
                    )
                )
            )


    # ========================================================
    # 6. NORMAL NAME CHARACTER N-GRAMS
    # ========================================================

    grams = name_ngrams(
        name_u
    )

    if grams:

        hashes = sorted(
            set(
                crc(g)
                for g in grams
            )
        )[:NGRAM_N]

        for h in hashes:

            out.append(
                (
                    "ng",
                    crc(
                        f"ng|{country}|{h}"
                    )
                )
            )


    # ========================================================
    # 7. TRANSLITERATED CHARACTER N-GRAMS
    #
    # Only added in the experimental version.
    # No exact transliterated-name key.
    # ========================================================

    if include_translit:

        translit_name = (
            transliterate_normalize(
                name_u
            )
        )

        ttoks = tokenize(
            translit_name
        )

        if ttoks:

            translit_clean = " ".join(
                t
                for t in ttoks
                if t not in LEGAL_SUFFIXES
            )

            if translit_clean:

                tgrams = name_ngrams(
                    translit_clean
                )

                if tgrams:

                    thashes = sorted(
                        set(
                            crc(g)
                            for g in tgrams
                        )
                    )[:NGRAM_N]

                    for h in thashes:

                        out.append(
                            (
                                "nt",
                                crc(
                                    f"nt|"
                                    f"{country}|"
                                    f"{h}"
                                )
                            )
                        )


    # ========================================================
    # ADDRESS
    # ========================================================

    ats = address_tokens(
        addr_u
    )

    useful = sorted(
        set(ats)
        - ADDRESS_GENERIC
    )

    nums = sorted(
        set(
            extract_numbers(
                addr_u
            )
        )
    )


    useful = useful[:4]
    nums = nums[:2]


    # ========================================================
    # ADDRESS TOKEN PAIRS
    # ========================================================

    for a, b in combinations(
        useful,
        2
    ):

        out.append(
            (
                "ap",
                crc(
                    f"ap|{country}|"
                    f"{a}|{b}"
                )
            )
        )


    # ========================================================
    # ADDRESS NUMBER + TOKEN
    # ========================================================

    for num in nums:

        for tok in useful:

            out.append(
                (
                    "an",
                    crc(
                        f"an|{country}|"
                        f"{num}|{tok}"
                    )
                )
            )


    return out


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(
    df
):

    df = df.copy()

    df["country_l"] = (
        df["country"]
        .fillna("")
        .str.strip()
        .str.casefold()
    )

    df["name_u"] = (
        df["business_name"]
        .map(normalize_name)
    )

    df["addr_u"] = (
        df["business_address"]
        .map(normalize_address)
    )

    return df


# ============================================================
# BUILD ADDRESS VOCAB
# ============================================================

def build_vocab(
    s1
):

    vocab = set()

    for r in s1.itertuples(
        index=False
    ):

        ats = address_tokens(
            r.addr_u
        )

        vocab.update(
            t
            for t in ats
            if t not in ADDRESS_GENERIC
        )

        vocab.update(
            extract_numbers(
                r.addr_u
            )
        )

    return vocab


# ============================================================
# BUILD BLOCKING INDEX
# ============================================================

def build_index(
    records,
    wanted_keys,
    include_translit
):

    index = defaultdict(
        list
    )

    counts = defaultdict(
        int
    )

    dead = set()

    record_ids = []


    for r in records.itertuples(
        index=False
    ):

        code = len(
            record_ids
        )

        record_ids.append(
            r.entity_id
        )


        keys = row_keys(
            r.country_l,
            r.name_u,
            r.addr_u,
            include_translit=include_translit
        )


        for sig, kh in keys:

            if kh not in wanted_keys:
                continue

            if kh in dead:
                continue


            counts[kh] += 1

            c = counts[kh]


            if c > KEY_CAP:

                dead.add(
                    kh
                )

                index.pop(
                    kh,
                    None
                )

                continue


            index[kh].append(
                code
            )


    return (
        dict(index),
        record_ids
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    s1,
    index,
    record_ids
):

    all_candidates = []

    per_s1 = []

    for i, r in enumerate(
        s1.itertuples(
            index=False
        )
    ):

        keys = row_keys(
            r.country_l,
            r.name_u,
            r.addr_u,
            include_translit=(
                "nt"
                in index["__SIGNAL_MARKER__"]
                if "__SIGNAL_MARKER__"
                in index
                else False
            )
        )

        # The marker above is only a convenience.
        # We actually determine nt mode from the caller below.

        seen = {}

        for sig, kh in keys:

            postings = index.get(
                kh
            )

            if not postings:
                continue

            for code in postings:

                seen[code] = (
                    seen.get(
                        code,
                        0
                    )
                    | (
                        1
                        << (
                            ALL_SIG_ID[sig]
                        )
                    )
                )

        all_candidates.append(
            set(
                seen.keys()
            )
        )

        per_s1.append(
            len(seen)
        )

    return (
        all_candidates,
        per_s1
    )


# ============================================================
# CANDIDATE GENERATOR WITH EXPLICIT MODE
# ============================================================

def generate_for_mode(
    s1,
    index,
    include_translit
):

    results = []

    counts = []

    for i, r in enumerate(
        s1.itertuples(
            index=False
        )
    ):

        keys = row_keys(
            r.country_l,
            r.name_u,
            r.addr_u,
            include_translit=include_translit
        )

        seen = set()

        for sig, kh in keys:

            postings = index.get(
                kh
            )

            if postings:

                seen.update(
                    postings
                )

        results.append(
            seen
        )

        counts.append(
            len(seen)
        )

    return (
        results,
        counts
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_truth(
    gt_path
):

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    truth = defaultdict(
        set
    )

    for r in gt.itertuples(
        index=False
    ):

        matches = str(
            r.matched_entity_ids
        ).strip()

        if not matches:
            continue

        truth[
            r.source1_entity_id
        ].update(
            x.strip()
            for x in matches.split(",")
            if x.strip()
        )

    return truth


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    s1,
    candidates,
    record_ids,
    truth
):

    s1_ids = (
        s1["entity_id"]
        .tolist()
    )

    recovered = 0
    total_true = 0

    per_s1_recall = []

    for i, cand_codes in enumerate(
        candidates
    ):

        sid = s1_ids[i]

        true_ids = truth.get(
            sid,
            set()
        )

        if not true_ids:
            continue

        total_true += len(
            true_ids
        )

        found_ids = {
            record_ids[c]
            for c in cand_codes
        }

        hit = len(
            true_ids
            & found_ids
        )

        recovered += hit

        per_s1_recall.append(
            hit / len(true_ids)
        )


    recall = (
        recovered / total_true
        if total_true
        else 0
    )


    return (
        total_true,
        recovered,
        recall,
        per_s1_recall
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TRANSLITERATION A/B TEST"
    )

    print(
        "5,000 S1 records"
    )

    print(
        "=" * 70
    )


    random.seed(
        RANDOM_SEED
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading S1..."
    )

    s1_full = pd.read_csv(
        os.path.join(
            TRAIN_ROOT,
            "train_source1.tsv"
        ),
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
        f"S1 total: "
        f"{len(s1_full):,}"
    )


    # ========================================================
    # SAMPLE
    # ========================================================

    sample_n = min(
        SAMPLE_SIZE,
        len(s1_full)
    )

    s1 = (
        s1_full
        .sample(
            n=sample_n,
            random_state=RANDOM_SEED
        )
        .copy()
    )

    s1 = prepare(
        s1
    )


    print(
        f"S1 sample: "
        f"{len(s1):,}"
    )


    # ========================================================
    # LOAD S2 + S3
    # ========================================================

    print(
        "\nLoading S2..."
    )

    s2 = pd.read_csv(
        os.path.join(
            TRAIN_ROOT,
            "train_source2.tsv"
        ),
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

    s2 = prepare(
        s2
    )


    print(
        f"S2: "
        f"{len(s2):,}"
    )


    print(
        "\nLoading S3..."
    )

    s3 = pd.read_csv(
        os.path.join(
            TRAIN_ROOT,
            "train_source3.tsv"
        ),
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

    s3 = prepare(
        s3
    )


    print(
        f"S3: "
        f"{len(s3):,}"
    )


    records = pd.concat(
        [
            s2,
            s3
        ],
        ignore_index=True
    )


    del s2
    del s3
    del s1_full


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print(
        "\nLoading ground truth..."
    )

    truth = load_truth(
        os.path.join(
            TRAIN_ROOT,
            "train_ground_truth.tsv"
        )
    )


    # ========================================================
    # KEEP ONLY S1 RECORDS WITH TRUTH
    # ========================================================

    sample_truth_count = sum(
        len(
            truth.get(
                sid,
                set()
            )
        )
        for sid in s1["entity_id"]
    )


    print(
        f"True pairs belonging to "
        f"5,000-row sample: "
        f"{sample_truth_count:,}"
    )


    # ========================================================
    # BUILD VOCAB
    # ========================================================

    print(
        "\nBuilding address vocabulary..."
    )

    vocab = build_vocab(
        s1
    )

    print(
        f"Vocabulary: "
        f"{len(vocab):,}"
    )


    # ========================================================
    # BUILD BASELINE WANTED KEYS
    # ========================================================

    print(
        "\nBuilding BASELINE S1 keys..."
    )

    base_s1_keys = []

    base_wanted = set()


    for r in s1.itertuples(
        index=False
    ):

        keys = row_keys(
            r.country_l,
            r.name_u,
            r.addr_u,
            include_translit=False
        )

        base_s1_keys.append(
            keys
        )

        base_wanted.update(
            kh
            for sig, kh in keys
        )


    print(
        f"Baseline wanted keys: "
        f"{len(base_wanted):,}"
    )


    # ========================================================
    # BUILD TRANSLIT WANTED KEYS
    # ========================================================

    print(
        "\nBuilding TRANSLITERATION S1 keys..."
    )

    translit_s1_keys = []

    translit_wanted = set()


    for r in s1.itertuples(
        index=False
    ):

        keys = row_keys(
            r.country_l,
            r.name_u,
            r.addr_u,
            include_translit=True
        )

        translit_s1_keys.append(
            keys
        )

        translit_wanted.update(
            kh
            for sig, kh in keys
        )


    print(
        f"Translit wanted keys: "
        f"{len(translit_wanted):,}"
    )


    # ========================================================
    # BUILD BASELINE INDEX
    # ========================================================

    print(
        "\nBuilding BASELINE index..."
    )

    base_index, base_record_ids = (
        build_index(
            records,
            base_wanted,
            include_translit=False
        )
    )


    print(
        f"Baseline live keys: "
        f"{len(base_index):,}"
    )


    # ========================================================
    # BUILD TRANSLIT INDEX
    # ========================================================

    print(
        "\nBuilding TRANSLITERATION index..."
    )

    translit_index, translit_record_ids = (
        build_index(
            records,
            translit_wanted,
            include_translit=True
        )
    )


    print(
        f"Translit live keys: "
        f"{len(translit_index):,}"
    )


    # ========================================================
    # GENERATE BASELINE CANDIDATES
    # ========================================================

    print(
        "\nGenerating BASELINE candidates..."
    )

    base_candidates = []

    base_counts = []


    for keys in base_s1_keys:

        seen = set()

        for sig, kh in keys:

            postings = base_index.get(
                kh
            )

            if postings:

                seen.update(
                    postings
                )

        base_candidates.append(
            seen
        )

        base_counts.append(
            len(seen)
        )


    # ========================================================
    # GENERATE TRANSLIT CANDIDATES
    # ========================================================

    print(
        "\nGenerating TRANSLITERATION candidates..."
    )

    translit_candidates = []

    translit_counts = []


    for keys in translit_s1_keys:

        seen = set()

        for sig, kh in keys:

            postings = translit_index.get(
                kh
            )

            if postings:

                seen.update(
                    postings
                )

        translit_candidates.append(
            seen
        )

        translit_counts.append(
            len(seen)
        )


    # ========================================================
    # EVALUATE BASELINE
    # ========================================================

    print(
        "\nEvaluating BASELINE..."
    )

    (
        base_true,
        base_recovered,
        base_recall,
        _
    ) = evaluate(
        s1,
        base_candidates,
        base_record_ids,
        truth
    )


    # ========================================================
    # EVALUATE TRANSLIT
    # ========================================================

    print(
        "\nEvaluating TRANSLITERATION..."
    )

    (
        trans_true,
        trans_recovered,
        trans_recall,
        _
    ) = evaluate(
        s1,
        translit_candidates,
        translit_record_ids,
        truth
    )


    # ========================================================
    # EXTRA RECOVERY
    # ========================================================

    extra_pairs = 0

    extra_true_recovered = 0


    for i in range(
        len(s1)
    ):

        base_set = (
            base_candidates[i]
        )

        trans_set = (
            translit_candidates[i]
        )

        extra_pairs += len(
            trans_set
            - base_set
        )


        sid = s1.iloc[i][
            "entity_id"
        ]

        true_ids = truth.get(
            sid,
            set()
        )

        if not true_ids:
            continue


        base_ids = {
            base_record_ids[c]
            for c in base_set
        }


        trans_ids = {
            translit_record_ids[c]
            for c in trans_set
        }


        extra_true_recovered += len(
            (
                trans_ids
                - base_ids
            )
            & true_ids
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    base_counts_s = pd.Series(
        base_counts
    )

    trans_counts_s = pd.Series(
        translit_counts
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL A/B RESULT"
    )

    print(
        "=" * 70
    )


    print(
        "\nBASELINE"
    )

    print(
        f"True pairs:       "
        f"{base_true:,}"
    )

    print(
        f"Recovered:        "
        f"{base_recovered:,}"
    )

    print(
        f"Recall:           "
        f"{base_recall:.4%}"
    )

    print(
        f"Mean candidates:  "
        f"{base_counts_s.mean():.1f}"
    )

    print(
        f"Median:           "
        f"{base_counts_s.median():.0f}"
    )

    print(
        f"P95:              "
        f"{base_counts_s.quantile(.95):.0f}"
    )

    print(
        f"P99:              "
        f"{base_counts_s.quantile(.99):.0f}"
    )

    print(
        f"Max:              "
        f"{base_counts_s.max():,.0f}"
    )


    print(
        "\n"
        + "-" * 70
    )


    print(
        "\nBASELINE + TRANSLITERATED N-GRAMS"
    )

    print(
        f"True pairs:       "
        f"{trans_true:,}"
    )

    print(
        f"Recovered:        "
        f"{trans_recovered:,}"
    )

    print(
        f"Recall:           "
        f"{trans_recall:.4%}"
    )

    print(
        f"Mean candidates:  "
        f"{trans_counts_s.mean():.1f}"
    )

    print(
        f"Median:           "
        f"{trans_counts_s.median():.0f}"
    )

    print(
        f"P95:              "
        f"{trans_counts_s.quantile(.95):.0f}"
    )

    print(
        f"P99:              "
        f"{trans_counts_s.quantile(.99):.0f}"
    )

    print(
        f"Max:              "
        f"{trans_counts_s.max():,.0f}"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRANSLITERATION LIFT"
    )

    print(
        "=" * 70
    )


    print(
        f"Recall improvement: "
        f"{trans_recall - base_recall:+.4%}"
    )

    print(
        f"Extra true pairs recovered: "
        f"{extra_true_recovered:,}"
    )

    print(
        f"Extra candidate pairs: "
        f"{extra_pairs:,}"
    )


    if base_counts_s.mean() > 0:

        candidate_growth = (
            trans_counts_s.mean()
            / base_counts_s.mean()
            - 1
        )

    else:

        candidate_growth = 0


    print(
        f"Mean candidate growth: "
        f"{candidate_growth:+.2%}"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "DECISION"
    )

    print(
        "=" * 70
    )


    lift = (
        trans_recall
        - base_recall
    )


    if lift >= 0.01:

        print(
            "KEEP nt:"
            " meaningful recall lift "
            "on the 5,000-row sample."
        )

    elif lift >= 0.005:

        print(
            "POSSIBLY KEEP nt:"
            " moderate recall lift."
        )

    else:

        print(
            "DROP nt:"
            " recall improvement is too small."
        )


    print(
        "\nExperiment complete."
    )


if __name__ == "__main__":

    main()