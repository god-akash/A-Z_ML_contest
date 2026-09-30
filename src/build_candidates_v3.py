"""
V3 HIGH-RECALL BLOCKER

Improvements over V2:
1. Keeps existing exact/token/address signals.
2. Stronger 3-character name n-gram signatures.
3. Non-Latin transliteration n-gram signal.
4. Postal-code + country signal.
5. Address character n-gram signal.
6. Stronger house-number + address-token signal.
7. Separate frequency caps by signal.
8. Streams candidates directly to Parquet.
9. Evaluates recall on TRAIN against full ground truth.

Run:
    python src\build_candidates_v3.py --split train

DO NOT run test until TRAIN recall is measured.
"""

import argparse
import os
import pickle
import re
import time
import unicodedata
import zlib
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


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

# ------------------------------------------------------------
# IMPORTANT:
# We use different caps for different signals.
# Exact signals can safely have a larger cap.
# Fuzzy character signals need a tighter cap.
# ------------------------------------------------------------

CAP_EXACT = 300
CAP_TOKEN = 300
CAP_NGRAM = 150
CAP_TRANSLIT = 150
CAP_ADDRESS_NGRAM = 100
CAP_POSTAL = 300
CAP_ADDRESS_STRONG = 200

BATCH_FLUSH_EVERY = 100_000


# ============================================================
# SIGNAL IDs
# ============================================================

SIGNALS = [
    "nx",      # exact normalized name
    "ns",      # sorted name tokens
    "nc",      # compact name
    "ua",      # unicode first/last
    "aa",      # ascii first/last

    "ng",      # stronger name character ngrams
    "nt",      # transliterated character ngrams

    "ap",      # address token pair
    "an",      # address number + token

    "pc",      # country + postal
    "ag",      # address character ngrams
    "ah",      # address number + strongest token
]

SIG_ID = {
    s: i
    for i, s in enumerate(SIGNALS)
}

CAPS = {
    "nx": CAP_EXACT,
    "ns": CAP_TOKEN,
    "nc": CAP_EXACT,
    "ua": CAP_EXACT,
    "aa": CAP_EXACT,

    "ng": CAP_NGRAM,
    "nt": CAP_TRANSLIT,

    "ap": CAP_ADDRESS_STRONG,
    "an": CAP_ADDRESS_STRONG,

    "pc": CAP_POSTAL,

    "ag": CAP_ADDRESS_NGRAM,
    "ah": CAP_ADDRESS_STRONG,
}


# ============================================================
# REGEX
# ============================================================

TOKEN_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)

NUM_RE = re.compile(
    r"\d+[a-zA-Z]?"
)

POSTAL_RE = re.compile(
    r"\b\d{4,6}\b"
)


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


# ============================================================
# HASH
# ============================================================

def crc(text):
    return zlib.crc32(
        text.encode("utf-8")
    )


# ============================================================
# NORMALIZATION
# ============================================================

def unicode_normalize(text):

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


def ascii_normalize(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text)
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
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


def transliterate_normalize(text):

    if not text:
        return ""

    s = str(text)

    # VERY IMPORTANT:
    # ASCII text does not need transliteration.
    if s.isascii():
        return ""

    if unidecode is None:
        return ""

    s = unidecode(s).casefold()

    s = re.sub(
        r"[^a-z0-9\s]",
        " ",
        s
    )

    return re.sub(
        r"\s+",
        " ",
        s
    ).strip()


def tokenize(text):

    if not text:
        return []

    return TOKEN_RE.findall(text)


def normalize_name(text):

    text = unicode_normalize(text)

    if not text:
        return ""

    text = TLD_RE.sub(
        "",
        text
    )

    tokens = tokenize(text)

    return " ".join(
        t
        for t in tokens
        if t not in LEGAL_SUFFIXES
    )


def normalize_address(text):

    return unicode_normalize(text)


# ============================================================
# BASIC EXTRACTION
# ============================================================

def extract_numbers(text):

    if not text:
        return []

    return NUM_RE.findall(text)


def postal_codes(text):

    if not text:
        return []

    return POSTAL_RE.findall(text)


def address_tokens(text):

    out = []

    for token in tokenize(text):

        if any(
            c.isdigit()
            for c in token
        ):
            out.append(token)

        elif len(token) >= 4:
            out.append(token)

    return out


# ============================================================
# CHARACTER NGRAMS
# ============================================================

def char_ngrams(
    text,
    k
):

    if not text:
        return []

    s = text.replace(
        " ",
        ""
    )

    if len(s) <= k:
        return [s] if s else []

    return [
        s[i:i+k]
        for i in range(
            len(s) - k + 1
        )
    ]


def smallest_hashes(
    grams,
    n
):

    if not grams:
        return []

    vals = {
        crc(g)
        for g in grams
    }

    return sorted(vals)[:n]


# ============================================================
# RARE/STRONG ADDRESS TOKEN SELECTION
# ============================================================

def strong_address_tokens(
    addr,
    vocab=None
):

    toks = address_tokens(addr)

    out = []

    for t in toks:

        if vocab is not None:

            if t not in vocab:
                continue

        if t in ADDRESS_GENERIC:
            continue

        # Prefer longer / more informative tokens.
        if len(t) >= 5:
            out.append(t)

    return out


# ============================================================
# ROW KEYS
# ============================================================

def row_keys(
    country,
    name_u,
    addr_u,
    vocab=None
):

    out = []

    ntoks = tokenize(
        name_u
    )

    # --------------------------------------------------------
    # 1. EXACT NORMALIZED NAME
    # --------------------------------------------------------

    if ntoks:

        ordered = " ".join(
            ntoks
        )

        out.append((
            SIG_ID["nx"],
            crc(
                f"nx|{country}|{ordered}"
            )
        ))

        # ----------------------------------------------------
        # 2. SORTED TOKENS
        # ----------------------------------------------------

        sorted_tokens = " ".join(
            sorted(
                set(ntoks)
            )
        )

        out.append((
            SIG_ID["ns"],
            crc(
                f"ns|{country}|{sorted_tokens}"
            )
        ))

        # ----------------------------------------------------
        # 3. COMPACT NAME
        # ----------------------------------------------------

        compact = "".join(
            ntoks
        )

        if len(compact) >= 6:

            out.append((
                SIG_ID["nc"],
                crc(
                    f"nc|{country}|{compact}"
                )
            ))

        # ----------------------------------------------------
        # 4. UNICODE FIRST/LAST
        # ----------------------------------------------------

        first = ntoks[0]
        last = ntoks[-1]

        if len(first) >= 2:

            out.append((
                SIG_ID["ua"],
                crc(
                    f"ua|{country}|"
                    f"{first[:5]}|"
                    f"{last[:5]}"
                )
            ))

    # --------------------------------------------------------
    # 5. ASCII FIRST/LAST
    # --------------------------------------------------------

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

            out.append((
                SIG_ID["aa"],
                crc(
                    f"aa|{country}|"
                    f"{first[:5]}|"
                    f"{last[:5]}"
                )
            ))

    # --------------------------------------------------------
    # 6. STRONGER NAME 3-GRAM SIGNATURE
    # --------------------------------------------------------

    grams3 = char_ngrams(
        name_u,
        3
    )

    for h in smallest_hashes(
        grams3,
        8
    ):

        out.append((
            SIG_ID["ng"],
            crc(
                f"ng|{country}|{h}"
            )
        ))

    # --------------------------------------------------------
    # 7. TRANSLITERATED NAME 3-GRAMS
    # --------------------------------------------------------

    trans = transliterate_normalize(
        name_u
    )

    if trans:

        trans_grams = char_ngrams(
            trans,
            3
        )

        for h in smallest_hashes(
            trans_grams,
            6
        ):

            out.append((
                SIG_ID["nt"],
                crc(
                    f"nt|{country}|{h}"
                )
            ))

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    ats = strong_address_tokens(
        addr_u,
        vocab=vocab
    )

    nums = extract_numbers(
        addr_u
    )

    # --------------------------------------------------------
    # 8. ADDRESS TOKEN PAIRS
    # --------------------------------------------------------

    if len(ats) >= 2:

        # Only a few combinations.
        # Prefer longer tokens.
        ats2 = sorted(
            set(ats),
            key=lambda x: (-len(x), x)
        )[:5]

        for i in range(
            min(len(ats2), 4)
        ):

            for j in range(
                i + 1,
                min(len(ats2), 5)
            ):

                out.append((
                    SIG_ID["ap"],
                    crc(
                        f"ap|{country}|"
                        f"{ats2[i]}|"
                        f"{ats2[j]}"
                    )
                ))

    # --------------------------------------------------------
    # 9. NUMBER + ADDRESS TOKEN
    # --------------------------------------------------------

    if nums and ats:

        for num in nums[:3]:

            for token in ats[:3]:

                out.append((
                    SIG_ID["an"],
                    crc(
                        f"an|{country}|"
                        f"{num}|{token}"
                    )
                ))

    # --------------------------------------------------------
    # 10. POSTAL CODE
    # --------------------------------------------------------

    for pc in postal_codes(
        addr_u
    ):

        out.append((
            SIG_ID["pc"],
            crc(
                f"pc|{country}|{pc}"
            )
        ))

    # --------------------------------------------------------
    # 11. ADDRESS CHARACTER NGRAMS
    # --------------------------------------------------------

    # Address character signal is intentionally small.
    # It is an auxiliary signal, not the main blocker.
    addr_compact = addr_u.replace(
        " ",
        ""
    )

    if len(addr_compact) >= 8:

        ag = char_ngrams(
            addr_u,
            4
        )

        for h in smallest_hashes(
            ag,
            4
        ):

            out.append((
                SIG_ID["ag"],
                crc(
                    f"ag|{country}|{h}"
                )
            ))

    # --------------------------------------------------------
    # 12. NUMBER + STRONGEST ADDRESS TOKEN
    # --------------------------------------------------------

    if nums and ats:

        strongest = sorted(
            set(ats),
            key=lambda x: (-len(x), x)
        )[:2]

        for num in nums[:2]:

            for token in strongest:

                out.append((
                    SIG_ID["ah"],
                    crc(
                        f"ah|{country}|"
                        f"{num}|{token}"
                    )
                ))

    return out


# ============================================================
# LOAD
# ============================================================

def load_source(
    split,
    source
):

    path = os.path.join(
        DATASET_ROOT,
        split,
        f"{split}_{source}.tsv"
    )

    return pd.read_csv(
        path,
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
# ADDRESS VOCAB
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

        for t in ats:

            if t not in ADDRESS_GENERIC:
                vocab.add(t)

        for n in extract_numbers(
            r.addr_u
        ):

            vocab.add(n)

    return vocab


# ============================================================
# S1 KEYS
# ============================================================

def build_s1_keys(
    s1
):

    s1_keys = []
    wanted = set()

    t0 = time.time()

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

        for _, kh in ks:
            wanted.add(kh)

        if (
            (i + 1)
            % BATCH_FLUSH_EVERY
            == 0
        ):

            rate = (
                (i + 1)
                /
                max(
                    time.time() - t0,
                    0.001
                )
            )

            print(
                f"  S1 keys "
                f"{i+1:,}/{len(s1):,} "
                f"rate={rate:.0f}/s "
                f"wanted={len(wanted):,}",
                flush=True
            )

    return (
        s1_keys,
        wanted
    )


# ============================================================
# INDEX
# ============================================================

def build_index(
    split,
    vocab,
    wanted
):

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
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ],
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

                for sig, kh in ks:

                    if kh not in wanted:
                        continue

                    if kh in dead:
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

                    count = (
                        counts.get(
                            kh,
                            0
                        )
                        + 1
                    )

                    counts[kh] = count

                    cap = CAPS[
                        SIGNALS[sig]
                    ]

                    if count > cap:

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
                    f"chunk={chunk_no} "
                    f"live_keys={len(index):,} "
                    f"dead={len(dead):,} "
                    f"records={len(rid_ids):,} "
                    f"t={time.time()-t0:.0f}s",
                    flush=True
                )

    print(
        "\nIndex complete:",
        len(index),
        "live keys",
        len(dead),
        "dead keys",
        len(rid_ids),
        "records"
    )

    return (
        index,
        rid_ids
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    s1_keys,
    index,
    out_path
):

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

    total = 0

    t0 = time.time()

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

            bit = (
                1 << sig
            )

            for code in postings:

                seen[code] = (
                    seen.get(
                        code,
                        0
                    )
                    |
                    bit
                )

        for code, mask in seen.items():

            buf_s1.append(i)

            buf_cand.append(
                code
            )

            buf_sig.append(
                mask.bit_count()
            )

        total += len(
            seen
        )

        if (
            (i + 1)
            % BATCH_FLUSH_EVERY
            == 0
        ):

            flush()

            rate = (
                (i + 1)
                /
                max(
                    time.time()-t0,
                    0.001
                )
            )

            print(
                f"  generated "
                f"{i+1:,}/{len(s1_keys):,} "
                f"S1 "
                f"rate={rate:.0f}/s "
                f"pairs={total:,}",
                flush=True
            )

    flush()

    writer.close()

    return total


# ============================================================
# TRAIN RECALL
# ============================================================

def evaluate_train_recall(
    out_path,
    s1_ids,
    rid_ids
):

    print(
        "\nEvaluating TRAIN recall..."
    )

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

    truth = set()

    for r in gt.itertuples(
        index=False
    ):

        s1i = s1_to_idx.get(
            r.source1_entity_id
        )

        if s1i is None:
            continue

        value = str(
            r.matched_entity_ids
        ).strip()

        if not value:
            continue

        for target in value.split(","):

            target = target.strip()

            code = rid_to_code.get(
                target
            )

            if code is not None:

                truth.add(
                    s1i * (
                        len(rid_ids)
                        + 1
                    )
                    + code
                )

    print(
        "Truth pairs:",
        len(truth)
    )

    recovered = 0

    per_s1 = defaultdict(int)

    pf = pq.ParquetFile(
        out_path
    )

    for batch in pf.iter_batches(
        batch_size=5_000_000
    ):

        s1_arr = batch[
            "s1_idx"
        ].to_numpy()

        cand_arr = batch[
            "cand_code"
        ].to_numpy()

        for x in s1_arr:
            per_s1[int(x)] += 1

        keys = (
            s1_arr.astype(
                "int64"
            )
            *
            (
                len(rid_ids)
                + 1
            )
            +
            cand_arr.astype(
                "int64"
            )
        )

        recovered += len(
            set(
                keys.tolist()
            )
            &
            truth
        )

    recall = (
        recovered
        /
        len(truth)
    )

    counts = pd.Series(
        list(
            per_s1.values()
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "V3 BLOCKING RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"TRUE PAIRS : {len(truth):,}"
    )

    print(
        f"RECOVERED  : {recovered:,}"
    )

    print(
        f"RECALL     : {recall:.6f}"
    )

    print(
        f"S1 WITH CANDIDATES: "
        f"{len(per_s1):,}/{len(s1_ids):,}"
    )

    print(
        f"MEAN CANDIDATES: "
        f"{counts.mean():.1f}"
    )

    print(
        f"MEDIAN: "
        f"{counts.median():.0f}"
    )

    print(
        f"P95: "
        f"{counts.quantile(.95):.0f}"
    )

    print(
        f"P99: "
        f"{counts.quantile(.99):.0f}"
    )

    print(
        f"MAX: "
        f"{counts.max():,}"
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

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    print(
        "\nLoading S1..."
    )

    s1 = prep(
        load_source(
            args.split,
            "source1"
        )
    )

    print(
        f"S1: {len(s1):,}"
    )

    # --------------------------------------------------------
    # ADDRESS VOCAB
    # --------------------------------------------------------

    print(
        "\nBuilding address vocabulary..."
    )

    vocab = build_vocab(
        s1
    )

    print(
        f"Address vocab: "
        f"{len(vocab):,}"
    )

    # --------------------------------------------------------
    # S1 KEYS
    # --------------------------------------------------------

    print(
        "\nBuilding S1 blocking keys..."
    )

    s1_keys, wanted = (
        build_s1_keys(
            s1
        )
    )

    s1_ids = [
        eid
        for eid, _
        in s1_keys
    ]

    print(
        f"S1 records: "
        f"{len(s1_ids):,}"
    )

    print(
        f"Wanted keys: "
        f"{len(wanted):,}"
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    print(
        "\nBuilding S2/S3 index..."
    )

    index, rid_ids = (
        build_index(
            args.split,
            vocab,
            wanted
        )
    )

    # --------------------------------------------------------
    # CACHE INDEX
    # --------------------------------------------------------

    with open(
        os.path.join(
            CACHE_DIR,
            f"{args.split}_v3_index.pkl"
        ),
        "wb"
    ) as f:

        pickle.dump(
            {
                "index": index,
                "rid_ids": rid_ids
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    with open(
        os.path.join(
            CACHE_DIR,
            f"{args.split}_v3_id_maps.pkl"
        ),
        "wb"
    ) as f:

        pickle.dump(
            {
                "s1_ids": s1_ids,
                "rid_ids": rid_ids
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    out_path = os.path.join(
        CACHE_DIR,
        f"{args.split}_v3_candidate_pairs.parquet"
    )

    print(
        "\nGenerating candidates..."
    )

    total = generate_candidates(
        s1_keys,
        index,
        out_path
    )

    print(
        f"\nCandidate pairs: "
        f"{total:,}"
    )

    # --------------------------------------------------------
    # TRAIN EVALUATION
    # --------------------------------------------------------

    if args.split == "train":

        evaluate_train_recall(
            out_path,
            s1_ids,
            rid_ids
        )

    print(
        "\nTotal time:",
        f"{time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()