# src/build_candidates.py

"""
Candidate generation (blocking) for the entity resolution pipeline.

Usage:
    python src/build_candidates.py --split train
    python src/build_candidates.py --split test

This version adds:
    nt = transliterated normalized business-name blocking signal

The original candidate/index files are preserved.

New files:
    cache/<split>_candidate_pairs_translit.parquet
    cache/<split>_id_maps_translit.pkl
    cache/<split>_index_translit.pkl
"""

import argparse
import os
import pickle
import re
import time
import unicodedata
import zlib

from collections import defaultdict
from itertools import combinations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

CACHE_DIR = os.path.join(
    ROOT,
    "cache"
)

CHUNK = 250_000

KEY_CAP = 300

NGRAM_K = 4
NGRAM_N = 4

BATCH_FLUSH_EVERY = 100_000


COLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]


# ============================================================
# BLOCKING SIGNALS
# ============================================================

SIGNALS = [
    "nx",   # exact normalized name
    "ns",   # sorted name tokens
    "nc",   # compact name
    "ap",   # address token pairs
    "an",   # address number + token
    "ua",   # unicode first/last token
    "aa",   # ascii first/last token
    "ng",   # character n-gram
    "nt"    # transliterated normalized name
]

SIG_ID = {
    s: i
    for i, s in enumerate(SIGNALS)
}


# ============================================================
# NORMALIZATION CONSTANTS
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
    """
    Stable hash.

    Python's hash() is randomized per process,
    so it must not be used for reusable blocking keys.
    """

    return zlib.crc32(
        key.encode("utf-8")
    )


# ============================================================
# NORMALIZATION
# ============================================================

def unicode_normalize(text) -> str:
    """
    Primary Unicode normalization.

    Keeps non-Latin scripts.
    """

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
    """
    ASCII-only auxiliary representation.

    This is NOT used as the primary name representation.
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text)
    )

    text = (
        text
        .encode("ascii", "ignore")
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
    """
    Transliterate non-Latin scripts into Latin characters.

    Example:

        ड्रीम केयर प्राइवेट लिमिटेड

    becomes approximately:

        dream care private limited

    This is used only as an auxiliary blocking signal.
    """

    if not text:
        return ""

    text = unidecode(
        str(text)
    )

    text = text.casefold()

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
# NAME N-GRAMS
# ============================================================

def name_ngrams(
    name_u: str,
    k: int = NGRAM_K
):

    """
    Character shingles.

    Spaces are removed.

    This helps with typos and partial corruption.
    """

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
# BLOCKING KEY GENERATION
# ============================================================

def row_keys(
    country: str,
    name_u: str,
    addr_u: str,
    vocab: set | None = None
):

    """
    Generate blocking keys for one record.

    S1:
        vocab=None

    S2/S3:
        vocab contains tokens/numbers that occur in S1.

    Every signal includes country so that records from
    different countries do not unnecessarily collide.
    """

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
                SIG_ID["nx"],
                crc(
                    f"nx|{country}|{' '.join(ntoks)}"
                )
            )
        )


        # ====================================================
        # 2. SORTED NAME TOKENS
        # ====================================================

        out.append(
            (
                SIG_ID["ns"],
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
                SIG_ID["nc"],
                crc(
                    f"nc|{country}|{compact}"
                )
            )
        )


    # ========================================================
    # 4. UNICODE FIRST/LAST TOKEN
    # ========================================================

    if ntoks:

        first = ntoks[0]
        last = ntoks[-1]

        if len(first) >= 2:

            out.append(
                (
                    SIG_ID["ua"],
                    crc(
                        f"ua|{country}|"
                        f"{first}|{last}"
                    )
                )
            )


    # ========================================================
    # 5. ASCII FIRST/LAST TOKEN
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
                    SIG_ID["aa"],
                    crc(
                        f"aa|{country}|"
                        f"{first}|{last}"
                    )
                )
            )


    # ========================================================
    # 6. TRANSLITERATED FULL NAME
    # ========================================================

    translit_name = transliterate_normalize(
        name_u
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

            out.append(
                (
                    SIG_ID["nt"],
                    crc(
                        f"nt|{country}|"
                        f"{translit_clean}"
                    )
                )
            )


    # ========================================================
    # 7. CHARACTER N-GRAM MINHASH
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
                    SIG_ID["ng"],
                    crc(
                        f"ng|{country}|{h}"
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
        set(ats) - ADDRESS_GENERIC
    )

    nums = sorted(
        set(
            extract_numbers(
                addr_u
            )
        )
    )


    # ========================================================
    # S2/S3 VOCAB PRUNING
    # ========================================================

    if vocab is not None:

        useful = [
            t
            for t in useful
            if t in vocab
        ]

        nums = [
            n
            for n in nums
            if n in vocab
        ]


    # ========================================================
    # LIMIT ADDRESS COMBINATIONS
    # ========================================================

    useful_capped = useful[:4]

    nums_capped = nums[:2]


    # ========================================================
    # 8. ADDRESS TOKEN PAIRS
    # ========================================================

    for a, b in combinations(
        useful_capped,
        2
    ):

        out.append(
            (
                SIG_ID["ap"],
                crc(
                    f"ap|{country}|"
                    f"{a}|{b}"
                )
            )
        )


    # ========================================================
    # 9. ADDRESS NUMBER + TOKEN
    # ========================================================

    for num in nums_capped:

        for tok in useful_capped:

            out.append(
                (
                    SIG_ID["an"],
                    crc(
                        f"an|{country}|"
                        f"{num}|{tok}"
                    )
                )
            )


    return out


# ============================================================
# LOAD SOURCE
# ============================================================

def load_source(
    split: str,
    source: str
):

    path = os.path.join(
        DATASET_ROOT,
        split,
        f"{split}_{source}.tsv"
    )

    print(
        f"Loading {path} ..."
    )

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=COLS,
        keep_default_na=False
    )


# ============================================================
# PREP
# ============================================================

def prep(df):

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
# BUILD ADDRESS VOCABULARY
# ============================================================

def build_vocab(
    s1: pd.DataFrame
) -> set:

    """
    Build every address token/number that can appear
    in an S1 key.
    """

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
# BUILD S1 KEYS
# ============================================================

def build_s1_keys(
    s1: pd.DataFrame
):

    s1_keys = []

    wanted = set()

    t0 = time.time()

    n = len(s1)

    for i, r in enumerate(
        s1.itertuples(
            index=False
        )
    ):

        ks = row_keys(
            r.country_l,
            r.name_u,
            r.addr_u,
            vocab=None
        )

        s1_keys.append(
            (
                r.entity_id,
                ks
            )
        )

        wanted.update(
            kh
            for _, kh in ks
        )

        if (
            (i + 1)
            % BATCH_FLUSH_EVERY
            == 0
        ):

            elapsed = (
                time.time()
                - t0
            )

            rate = (
                (i + 1)
                / elapsed
            )

            print(
                f"  S1 keys "
                f"{i + 1:,}/{n:,} "
                f"({rate:.0f} rec/s) "
                f"keys={len(wanted):,}",
                flush=True
            )

    return (
        s1_keys,
        wanted
    )


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    split: str,
    vocab: set,
    wanted: set
):

    """
    Build the S2/S3 blocking index.

    KEY_CAP protects against huge generic buckets.

    A key with > KEY_CAP records is removed entirely.
    """

    index = {}

    counts = {}

    dead = set()

    rid_ids = []

    rid_map = {}

    t0 = time.time()


    for source in [
        "source2",
        "source3"
    ]:

        path = os.path.join(
            DATASET_ROOT,
            split,
            f"{split}_{source}.tsv"
        )

        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=COLS,
            chunksize=CHUNK,
            keep_default_na=False
        )


        for chunk_no, chunk in enumerate(
            reader
        ):

            chunk = prep(
                chunk
            )


            for r in chunk.itertuples(
                index=False
            ):

                ks = row_keys(
                    r.country_l,
                    r.name_u,
                    r.addr_u,
                    vocab=vocab
                )

                code = None


                for _, kh in ks:

                    if (
                        kh not in wanted
                        or kh in dead
                    ):
                        continue


                    if code is None:

                        code = rid_map.get(
                            r.entity_id
                        )

                        if code is None:

                            code = len(
                                rid_ids
                            )

                            rid_map[
                                r.entity_id
                            ] = code

                            rid_ids.append(
                                r.entity_id
                            )


                    c = (
                        counts.get(
                            kh,
                            0
                        )
                        + 1
                    )

                    counts[
                        kh
                    ] = c


                    if c > KEY_CAP:

                        dead.add(
                            kh
                        )

                        index.pop(
                            kh,
                            None
                        )

                        continue


                    index.setdefault(
                        kh,
                        []
                    ).append(
                        code
                    )


            if chunk_no % 5 == 0:

                print(
                    f"  {source} "
                    f"chunk {chunk_no}: "
                    f"index_keys="
                    f"{len(index):,} "
                    f"dead_keys="
                    f"{len(dead):,} "
                    f"rids_seen="
                    f"{len(rid_ids):,} "
                    f"t="
                    f"{time.time() - t0:.0f}s",
                    flush=True
                )


    print(
        f"\nIndex built: "
        f"{len(index):,} live keys, "
        f"{len(dead):,} dead keys, "
        f"{len(rid_ids):,} records referenced. "
        f"t={time.time() - t0:.0f}s"
    )

    return (
        index,
        rid_ids
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    s1_keys,
    index,
    out_path,
    checkpoint_every=BATCH_FLUSH_EVERY
):

    """
    Stream candidate pairs directly to Parquet.

    Each candidate is stored using integer codes:

        s1_idx
        cand_code
        n_signals
    """

    schema = pa.schema([
        (
            "s1_idx",
            pa.int32()
        ),
        (
            "cand_code",
            pa.int32()
        ),
        (
            "n_signals",
            pa.int8()
        ),
    ])


    writer = pq.ParquetWriter(
        out_path,
        schema
    )


    buf_s1 = []
    buf_cand = []
    buf_sig = []


    t0 = time.time()

    total_pairs = 0

    n = len(
        s1_keys
    )


    def flush():

        nonlocal \
            buf_s1, \
            buf_cand, \
            buf_sig


        if not buf_s1:
            return


        batch = pa.record_batch(
            [
                pa.array(
                    buf_s1,
                    type=pa.int32()
                ),

                pa.array(
                    buf_cand,
                    type=pa.int32()
                ),

                pa.array(
                    buf_sig,
                    type=pa.int8()
                ),
            ],
            schema=schema
        )


        writer.write_batch(
            batch
        )


        buf_s1 = []
        buf_cand = []
        buf_sig = []


    for i, (
        eid,
        ks
    ) in enumerate(
        s1_keys
    ):

        seen = {}


        for sig, kh in ks:

            postings = index.get(
                kh
            )

            if not postings:
                continue


            bit = 1 << sig


            for code in postings:

                seen[code] = (
                    seen.get(
                        code,
                        0
                    )
                    | bit
                )


        for code, mask in seen.items():

            buf_s1.append(
                i
            )

            buf_cand.append(
                code
            )

            buf_sig.append(
                mask.bit_count()
            )


        total_pairs += len(
            seen
        )


        if (
            (i + 1)
            % checkpoint_every
            == 0
        ):

            flush()


            elapsed = (
                time.time()
                - t0
            )

            rate = (
                (i + 1)
                / elapsed
            )

            remaining = (
                n - (i + 1)
            ) / rate


            print(
                f"  generated "
                f"{i + 1:,}/{n:,} S1 "
                f"({rate:.0f} rec/s, "
                f"~{remaining / 60:.1f} "
                f"min remaining, "
                f"{total_pairs:,} "
                f"pair rows so far)",
                flush=True
            )


    flush()

    writer.close()

    return total_pairs


# ============================================================
# STATS + RECALL
# ============================================================

def compute_stats_and_recall(
    split,
    out_path,
    s1_ids,
    rid_ids
):

    print(
        "\nComputing candidate stats"
        + (
            " + recall"
            if split == "train"
            else ""
        )
        + " (streaming)..."
    )


    truth_codes = set()

    n_true_total = 0


    if split == "train":

        gt_path = os.path.join(
            DATASET_ROOT,
            "train",
            "train_ground_truth.tsv"
        )


        gt = pd.read_csv(
            gt_path,
            sep="\t",
            dtype=str,
            keep_default_na=False
        )


        s1_id_to_idx = {
            eid: i
            for i, eid
            in enumerate(
                s1_ids
            )
        }


        rid_to_code = {
            rid: c
            for c, rid
            in enumerate(
                rid_ids
            )
        }


        for r in gt.itertuples(
            index=False
        ):

            matches = str(
                r.matched_entity_ids
            ).strip()


            if not matches:
                continue


            n_true_total += len(
                matches.split(",")
            )


            s1i = s1_id_to_idx.get(
                r.source1_entity_id
            )


            if s1i is None:
                continue


            for other in matches.split(","):

                other = other.strip()

                c = rid_to_code.get(
                    other
                )


                if c is not None:

                    truth_codes.add(
                        (
                            s1i,
                            c
                        )
                    )


    per_s1_counts = defaultdict(
        int
    )

    recovered = 0


    pf = pq.ParquetFile(
        out_path
    )


    for batch in pf.iter_batches(
        batch_size=5_000_000
    ):

        s1_arr = (
            batch
            .column("s1_idx")
            .to_numpy()
        )

        cand_arr = (
            batch
            .column("cand_code")
            .to_numpy()
        )


        for a in s1_arr:

            per_s1_counts[
                a
            ] += 1


        if (
            split == "train"
            and truth_codes
        ):

            pairs = set(
                zip(
                    s1_arr.tolist(),
                    cand_arr.tolist()
                )
            )


            recovered += len(
                pairs
                & truth_codes
            )


    counts = (
        pd.Series(
            list(
                per_s1_counts.values()
            )
        )
        if per_s1_counts
        else
        pd.Series([0])
    )


    print(
        f"\nS1 with candidates: "
        f"{len(per_s1_counts):,} / "
        f"{len(s1_ids):,}"
    )


    print(
        f"Mean: "
        f"{counts.mean():.1f}  "
        f"Median: "
        f"{counts.median():.0f}  "
        f"P95: "
        f"{counts.quantile(.95):.0f}  "
        f"P99: "
        f"{counts.quantile(.99):.0f}  "
        f"Max: "
        f"{counts.max():,}"
    )


    if split == "train":

        if n_true_total:

            print(
                f"\nTRUE PAIRS: "
                f"{n_true_total:,}   "
                f"RECOVERED: "
                f"{recovered:,}   "
                f"RECALL: "
                f"{recovered / n_true_total:.4f}"
            )

        else:

            print(
                "No ground-truth pairs found."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test"
        ],
        required=True
    )

    args = parser.parse_args()


    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )


    t0 = time.time()


    # ========================================================
    # LOAD S1
    # ========================================================

    s1 = prep(
        load_source(
            args.split,
            "source1"
        )
    )


    print(
        f"S1 loaded: "
        f"{len(s1):,}  "
        f"t={time.time() - t0:.0f}s"
    )


    # ========================================================
    # VOCAB
    # ========================================================

    print(
        "\nBuilding address vocabulary..."
    )


    vocab = build_vocab(
        s1
    )


    print(
        f"Vocab size: "
        f"{len(vocab):,} "
        f"t={time.time() - t0:.0f}s"
    )


    # ========================================================
    # S1 KEYS
    # ========================================================

    print(
        "\nBuilding S1 keys..."
    )


    s1_keys, wanted = build_s1_keys(
        s1
    )


    s1_ids = [
        eid
        for eid, _
        in s1_keys
    ]


    print(
        f"S1 key rows: "
        f"{len(s1_keys):,}  "
        f"wanted unique keys: "
        f"{len(wanted):,}  "
        f"t={time.time() - t0:.0f}s"
    )


    # ========================================================
    # INDEX
    # ========================================================

    index, rid_ids = build_index(
        args.split,
        vocab,
        wanted
    )


    # ========================================================
    # IMPORTANT:
    # NEW EXPERIMENT FILES
    # ========================================================

    idx_cache_path = os.path.join(
        CACHE_DIR,
        f"{args.split}_index_translit.pkl"
    )


    with open(
        idx_cache_path,
        "wb"
    ) as f:

        pickle.dump(
            {
                "index": index,
                "rid_ids": rid_ids
            },
            f
        )


    print(
        f"Index cached to "
        f"{idx_cache_path} "
        f"t={time.time() - t0:.0f}s"
    )


    # ========================================================
    # ID MAP
    # ========================================================

    id_map_path = os.path.join(
        CACHE_DIR,
        f"{args.split}_id_maps_translit.pkl"
    )


    with open(
        id_map_path,
        "wb"
    ) as f:

        pickle.dump(
            {
                "s1_ids": s1_ids,
                "rid_ids": rid_ids
            },
            f
        )


    # ========================================================
    # GENERATE CANDIDATES
    # ========================================================

    print(
        "\nGenerating candidate pairs..."
    )


    out_path = os.path.join(
        CACHE_DIR,
        f"{args.split}_candidate_pairs_translit.parquet"
    )


    total_pairs = generate_candidates(
        s1_keys,
        index,
        out_path
    )


    print(
        f"Candidate pairs: "
        f"{total_pairs:,} "
        f"t={time.time() - t0:.0f}s"
    )


    print(
        f"Saved: "
        f"{out_path}"
    )


    # ========================================================
    # RECALL
    # ========================================================

    compute_stats_and_recall(
        args.split,
        out_path,
        s1_ids,
        rid_ids
    )


    # ========================================================
    # DONE
    # ========================================================

    print(
        f"\nTotal time: "
        f"{time.time() - t0:.0f}s"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()