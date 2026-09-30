"""
V4 HIGH-RECALL BLOCKER

Main V4 changes over V3
-----------------------
1. Rare character 3-gram name blocking instead of "smallest hash" selection.
2. Multiple informative name 3-gram keys per S1 record.
3. Explicit Latin <-> transliteration bridge.
4. Rare address-token blocking.
5. Keeps exact/token/address/number/postal signals.
6. Separate frequency caps.
7. Streams candidate pairs to Parquet.
8. Full ground-truth recall evaluation.

Run TRAIN first:

    python src\build_candidates_v4.py --split train

Only after checking recall:

    python src\build_candidates_v4.py --split test
"""

import argparse
import os
import pickle
import re
import time
import unicodedata
import zlib
from collections import defaultdict, Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


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

CHUNK = 250_000


# ============================================================
# CAPS
# ============================================================

CAP_EXACT = 300
CAP_TOKEN = 300

# Name fuzzy keys
CAP_NAME_NGRAM = 150
CAP_TRANSLIT = 150

# Address keys
CAP_ADDRESS_PAIR = 200
CAP_ADDRESS_RARE = 150
CAP_ADDRESS_NUMBER = 200
CAP_POSTAL = 300
CAP_ADDRESS_NGRAM = 100


# ============================================================
# SIGNALS
# ============================================================

SIGNALS = [
    "nx",      # exact normalized name
    "ns",      # sorted name tokens
    "nc",      # compact name
    "ua",      # unicode first/last
    "aa",      # ascii first/last

    "ng",      # rare name character 3-gram
    "nt",      # Latin <-> transliteration bridge

    "ap",      # address token pair
    "ar",      # rare address token
    "an",      # address number + token

    "pc",      # country + postal
    "ag",      # address character ngram
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

    "ng": CAP_NAME_NGRAM,
    "nt": CAP_TRANSLIT,

    "ap": CAP_ADDRESS_PAIR,
    "ar": CAP_ADDRESS_RARE,
    "an": CAP_ADDRESS_NUMBER,

    "pc": CAP_POSTAL,
    "ag": CAP_ADDRESS_NGRAM,
    "ah": CAP_ADDRESS_NUMBER,
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

TLD_RE = re.compile(
    r"\.(com|net|org|info|biz|co\.in|in|us|fr)\b",
    flags=re.IGNORECASE
)


# ============================================================
# CONSTANTS
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
    "near",
    "opposite",
    "main",
    "north",
    "south",
    "east",
    "west",
})


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
        token
        for token in tokenize(text)
        if token not in LEGAL_SUFFIXES
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


def postal_codes(text):

    if not text:
        return []

    return POSTAL_RE.findall(
        text
    )


# ============================================================
# CHARACTER NGRAMS
# ============================================================

def char_ngrams(text, n=3):

    if not text:
        return []

    # Remove spaces so:
    #
    # "great software"
    #
    # becomes:
    #
    # "greatsoftware"
    #
    s = text.replace(
        " ",
        ""
    )

    if len(s) < n:
        return [s]

    return [
        s[i:i+n]
        for i in range(
            len(s) - n + 1
        )
    ]


# ============================================================
# TRANSLITERATION
# ============================================================

def transliterate_normalize(text):

    if not text:
        return ""

    # ASCII names already have the correct Latin representation.
    if text.isascii():
        return text

    if unidecode is None:
        return ""

    try:
        t = unidecode(
            text
        )
    except Exception:
        return ""

    return ascii_normalize(
        t
    )


# ============================================================
# ADDRESS TOKENS
# ============================================================

def address_tokens(text):

    if not text:
        return []

    out = []

    for token in tokenize(text):

        if token in ADDRESS_GENERIC:
            continue

        if any(
            c.isdigit()
            for c in token
        ):
            out.append(
                token
            )

        elif len(token) >= 4:
            out.append(
                token
            )

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


def prepare_dataframe(df):

    df = df.copy()

    df["country_l"] = (
        df["country"]
        .fillna("")
        .str.strip()
        .str.casefold()
    )

    df["name_u"] = (
        df["business_name"]
        .map(
            normalize_name
        )
    )

    df["addr_u"] = (
        df["business_address"]
        .map(
            normalize_address
        )
    )

    return df


# ============================================================
# S1 STATISTICS
# ============================================================

def build_name_ngram_df(
    s1
):

    """
    Build country-specific document frequency of
    normalized Latin name 3-grams.

    This is calculated ONLY from S1.

    Why?

    We want informative grams.

    Example:

        "the" -> probably common
        "xyz" -> probably rare

    Rare grams are much better blocking keys.
    """

    print()
    print(
        "Building S1 name 3-gram frequencies..."
    )

    counters = defaultdict(
        Counter
    )

    for i, row in enumerate(
        s1.itertuples(
            index=False
        )
    ):

        name = row.name_u

        if not name:
            continue

        grams = set(
            char_ngrams(
                name,
                3
            )
        )

        country = row.country_l

        counters[
            country
        ].update(
            grams
        )

        if (
            i + 1
        ) % 250_000 == 0:

            print(
                f"  S1 name stats: "
                f"{i+1:,}"
            )

    return counters


def build_address_token_df(
    s1
):

    print()
    print(
        "Building S1 address-token frequencies..."
    )

    counters = defaultdict(
        Counter
    )

    for i, row in enumerate(
        s1.itertuples(
            index=False
        )
    ):

        tokens = set(
            address_tokens(
                row.addr_u
            )
        )

        if not tokens:
            continue

        counters[
            row.country_l
        ].update(
            tokens
        )

        if (
            i + 1
        ) % 250_000 == 0:

            print(
                f"  S1 address stats: "
                f"{i+1:,}"
            )

    return counters


# ============================================================
# SELECT RARE NAME GRAMS
# ============================================================

def select_name_grams(
    name,
    country,
    name_df
):

    grams = list(
        set(
            char_ngrams(
                name,
                3
            )
        )
    )

    if not grams:
        return []


    counter = name_df.get(
        country,
        {}
    )


    # --------------------------------------------------------
    # First preference:
    # rare grams
    # --------------------------------------------------------

    rare = []

    for g in grams:

        freq = counter.get(
            g,
            0
        )

        if (
            freq > 0
            and freq <= 500
        ):
            rare.append(
                (
                    freq,
                    g
                )
            )


    rare.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )


    selected = [
        g
        for _, g in rare[:8]
    ]


    # --------------------------------------------------------
    # If not enough rare grams,
    # add moderately frequent grams.
    # --------------------------------------------------------

    if len(selected) < 5:

        fallback = []

        for g in grams:

            if g in selected:
                continue

            freq = counter.get(
                g,
                0
            )

            if (
                freq > 0
                and freq <= 5000
            ):
                fallback.append(
                    (
                        freq,
                        g
                    )
                )

        fallback.sort(
            key=lambda x: (
                x[0],
                x[1]
            )
        )

        for _, g in fallback:

            if g not in selected:

                selected.append(
                    g
                )

            if len(selected) >= 8:
                break


    return selected[:8]


# ============================================================
# SELECT RARE ADDRESS TOKENS
# ============================================================

def select_address_tokens(
    addr,
    country,
    address_df
):

    tokens = list(
        set(
            address_tokens(
                addr
            )
        )
    )

    if not tokens:
        return []


    counter = address_df.get(
        country,
        {}
    )


    candidates = []

    for token in tokens:

        freq = counter.get(
            token,
            0
        )

        # Avoid extremely common address words.
        if (
            freq > 0
            and freq <= 1000
        ):
            candidates.append(
                (
                    freq,
                    token
                )
            )


    candidates.sort(
        key=lambda x: (
            x[0],
            -len(x[1]),
            x[1]
        )
    )


    return [
        token
        for _, token in candidates[:3]
    ]


# ============================================================
# S1 KEY GENERATION
# ============================================================

def build_s1_keys(
    s1,
    name_df,
    address_df
):

    print()
    print(
        "Building S1 blocking keys..."
    )

    s1_keys = []

    wanted = set()

    t0 = time.time()

    for i, row in enumerate(
        s1.itertuples(
            index=False
        )
    ):

        country = row.country_l
        name = row.name_u
        addr = row.addr_u

        keys = []


        # ====================================================
        # 1. EXACT NAME
        # ====================================================

        tokens = tokenize(
            name
        )

        if tokens:

            key = crc(
                f"nx|{country}|{' '.join(tokens)}"
            )

            keys.append(
                (
                    SIG_ID["nx"],
                    key
                )
            )


            # =================================================
            # 2. SORTED TOKENS
            # =================================================

            sorted_tokens = " ".join(
                sorted(
                    set(tokens)
                )
            )

            keys.append(
                (
                    SIG_ID["ns"],
                    crc(
                        f"ns|{country}|{sorted_tokens}"
                    )
                )
            )


            # =================================================
            # 3. COMPACT NAME
            # =================================================

            compact = "".join(
                tokens
            )

            if len(compact) >= 6:

                keys.append(
                    (
                        SIG_ID["nc"],
                        crc(
                            f"nc|{country}|{compact}"
                        )
                    )
                )


            # =================================================
            # 4. UNICODE FIRST/LAST
            # =================================================

            first = tokens[0]
            last = tokens[-1]

            if len(first) >= 2:

                keys.append(
                    (
                        SIG_ID["ua"],
                        crc(
                            f"ua|{country}|"
                            f"{first}|{last}"
                        )
                    )
                )


        # ====================================================
        # 5. ASCII FIRST/LAST
        # ====================================================

        ascii_name = ascii_normalize(
            name
        )

        ascii_tokens = tokenize(
            ascii_name
        )

        if ascii_tokens:

            first = ascii_tokens[0]
            last = ascii_tokens[-1]

            if len(first) >= 2:

                keys.append(
                    (
                        SIG_ID["aa"],
                        crc(
                            f"aa|{country}|"
                            f"{first}|{last}"
                        )
                    )
                )


        # ====================================================
        # 6. RARE NAME 3-GRAM
        # ====================================================

        selected_grams = select_name_grams(
            name,
            country,
            name_df
        )

        for gram in selected_grams:

            keys.append(
                (
                    SIG_ID["ng"],
                    crc(
                        f"ng|{country}|{gram}"
                    )
                )
            )


        # ====================================================
        # 7. TRANSLITERATION BRIDGE
        #
        # IMPORTANT:
        #
        # S1 English:
        #     "great software"
        #
        # Target Hindi:
        #     Hindi-script version
        #
        # Target is transliterated into Latin.
        #
        # Both are compared through the SAME namespace.
        # ====================================================

        if name:

            # For Latin S1 this is simply the Latin name.
            bridge_name = (
                transliterate_normalize(
                    name
                )
                if not name.isascii()
                else name
            )

            if bridge_name:

                bridge_grams = select_name_grams(
                    bridge_name,
                    country,
                    name_df
                )

                for gram in bridge_grams:

                    keys.append(
                        (
                            SIG_ID["nt"],
                            crc(
                                f"nt|{country}|{gram}"
                            )
                        )
                    )


        # ====================================================
        # ADDRESS
        # ====================================================

        nums = extract_numbers(
            addr
        )

        ats = address_tokens(
            addr
        )


        # ====================================================
        # 8. ADDRESS TOKEN PAIRS
        # ====================================================

        strong_ats = sorted(
            set(ats),
            key=lambda x: (
                -len(x),
                x
            )
        )[:5]

        if len(strong_ats) >= 2:

            for a in range(
                len(strong_ats)
            ):

                for b in range(
                    a + 1,
                    len(strong_ats)
                ):

                    token_a = strong_ats[a]
                    token_b = strong_ats[b]

                    pair = "|".join(
                        sorted(
                            [
                                token_a,
                                token_b
                            ]
                        )
                    )

                    keys.append(
                        (
                            SIG_ID["ap"],
                            crc(
                                f"ap|{country}|{pair}"
                            )
                        )
                    )


        # ====================================================
        # 9. RARE ADDRESS TOKEN
        # ====================================================

        rare_address = select_address_tokens(
            addr,
            country,
            address_df
        )

        for token in rare_address:

            keys.append(
                (
                    SIG_ID["ar"],
                    crc(
                        f"ar|{country}|{token}"
                    )
                )
            )


        # ====================================================
        # 10. NUMBER + ADDRESS TOKEN
        # ====================================================

        if nums and ats:

            for number in nums[:3]:

                for token in strong_ats[:3]:

                    keys.append(
                        (
                            SIG_ID["an"],
                            crc(
                                f"an|{country}|"
                                f"{number}|{token}"
                            )
                        )
                    )


        # ====================================================
        # 11. POSTAL
        # ====================================================

        for postal in postal_codes(
            addr
        ):

            keys.append(
                (
                    SIG_ID["pc"],
                    crc(
                        f"pc|{country}|{postal}"
                    )
                )
            )


        # ====================================================
        # 12. ADDRESS CHAR NGRAM
        # ====================================================

        compact_addr = addr.replace(
            " ",
            ""
        )

        if len(compact_addr) >= 8:

            grams = char_ngrams(
                addr,
                4
            )

            # Use a small deterministic subset.
            grams = sorted(
                set(grams)
            )[:4]

            for gram in grams:

                keys.append(
                    (
                        SIG_ID["ag"],
                        crc(
                            f"ag|{country}|{gram}"
                        )
                    )
                )


        # ====================================================
        # 13. NUMBER + STRONG ADDRESS TOKEN
        # ====================================================

        if nums and strong_ats:

            for number in nums[:2]:

                for token in strong_ats[:2]:

                    keys.append(
                        (
                            SIG_ID["ah"],
                            crc(
                                f"ah|{country}|"
                                f"{number}|{token}"
                            )
                        )
                    )


        # Remove duplicate keys
        keys = list(
            set(keys)
        )

        s1_keys.append(
            (
                row.entity_id,
                keys
            )
        )

        for signal_id, key_hash in keys:

            wanted.add(
                (
                    signal_id,
                    key_hash
                )
            )


        if (
            i + 1
        ) % 250_000 == 0:

            print(
                f"  S1 keys: "
                f"{i+1:,} / "
                f"{len(s1):,} | "
                f"wanted={len(wanted):,} | "
                f"time={time.time()-t0:.1f}s"
            )


    print()
    print(
        f"S1 entities: {len(s1):,}"
    )

    print(
        f"Wanted keys: {len(wanted):,}"
    )

    return (
        s1_keys,
        wanted
    )


# ============================================================
# TARGET KEY GENERATION
# ============================================================

def target_keys(
    country,
    name,
    addr
):

    out = []

    tokens = tokenize(
        name
    )


    # ========================================================
    # EXACT NAME
    # ========================================================

    if tokens:

        out.append(
            (
                SIG_ID["nx"],
                crc(
                    f"nx|{country}|{' '.join(tokens)}"
                )
            )
        )


        sorted_tokens = " ".join(
            sorted(
                set(tokens)
            )
        )

        out.append(
            (
                SIG_ID["ns"],
                crc(
                    f"ns|{country}|{sorted_tokens}"
                )
            )
        )


        compact = "".join(
            tokens
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


        first = tokens[0]
        last = tokens[-1]

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
    # ASCII FIRST/LAST
    # ========================================================

    ascii_name = ascii_normalize(
        name
    )

    ascii_tokens = tokenize(
        ascii_name
    )

    if ascii_tokens:

        first = ascii_tokens[0]
        last = ascii_tokens[-1]

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
    # NORMAL NAME NGRAMS
    # ========================================================

    grams = char_ngrams(
        name,
        3
    )

    # IMPORTANT:
    # We cannot use the S1 frequency table here.
    # The target side simply generates all grams.
    #
    # The index will only contain grams selected from S1.

    for gram in set(
        grams
    ):

        out.append(
            (
                SIG_ID["ng"],
                crc(
                    f"ng|{country}|{gram}"
                )
            )
        )


    # ========================================================
    # TRANSLITERATION BRIDGE
    #
    # This is the major V4 fix.
    #
    # English S1:
    #     "great software"
    #
    # Hindi S2:
    #     Hindi text
    #
    # Target Hindi is transliterated to Latin.
    #
    # Then target transliteration uses "nt".
    # S1 Latin name ALSO uses "nt".
    # ========================================================

    bridge_name = (
        transliterate_normalize(
            name
        )
    )

    if bridge_name:

        trans_grams = char_ngrams(
            bridge_name,
            3
        )

        for gram in set(
            trans_grams
        ):

            out.append(
                (
                    SIG_ID["nt"],
                    crc(
                        f"nt|{country}|{gram}"
                    )
                )
            )


    # ========================================================
    # ADDRESS
    # ========================================================

    nums = extract_numbers(
        addr
    )

    ats = address_tokens(
        addr
    )

    strong_ats = sorted(
        set(ats),
        key=lambda x: (
            -len(x),
            x
        )
    )[:5]


    # ========================================================
    # ADDRESS PAIRS
    # ========================================================

    if len(strong_ats) >= 2:

        for a in range(
            len(strong_ats)
        ):

            for b in range(
                a + 1,
                len(strong_ats)
            ):

                pair = "|".join(
                    sorted(
                        [
                            strong_ats[a],
                            strong_ats[b]
                        ]
                    )
                )

                out.append(
                    (
                        SIG_ID["ap"],
                        crc(
                            f"ap|{country}|{pair}"
                        )
                    )
                )


    # ========================================================
    # ADDRESS TOKEN
    # ========================================================

    for token in strong_ats[:3]:

        out.append(
            (
                SIG_ID["ar"],
                crc(
                    f"ar|{country}|{token}"
                )
            )
        )


    # ========================================================
    # NUMBER + TOKEN
    # ========================================================

    if nums and strong_ats:

        for number in nums[:3]:

            for token in strong_ats[:3]:

                out.append(
                    (
                        SIG_ID["an"],
                        crc(
                            f"an|{country}|"
                            f"{number}|{token}"
                        )
                    )
                )


    # ========================================================
    # POSTAL
    # ========================================================

    for postal in postal_codes(
        addr
    ):

        out.append(
            (
                SIG_ID["pc"],
                crc(
                    f"pc|{country}|{postal}"
                )
            )
        )


    # ========================================================
    # ADDRESS NGRAM
    # ========================================================

    compact_addr = addr.replace(
        " ",
        ""
    )

    if len(compact_addr) >= 8:

        grams = char_ngrams(
            addr,
            4
        )

        for gram in sorted(
            set(grams)
        )[:4]:

            out.append(
                (
                    SIG_ID["ag"],
                    crc(
                        f"ag|{country}|{gram}"
                    )
                )
            )


    # ========================================================
    # NUMBER + STRONG ADDRESS
    # ========================================================

    if nums and strong_ats:

        for number in nums[:2]:

            for token in strong_ats[:2]:

                out.append(
                    (
                        SIG_ID["ah"],
                        crc(
                            f"ah|{country}|"
                            f"{number}|{token}"
                        )
                    )
                )


    return list(
        set(out)
    )


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    split,
    wanted
):

    print()
    print(
        "=" * 70
    )

    print(
        "BUILDING TARGET INDEX"
    )

    print(
        "=" * 70
    )

    index = defaultdict(
        list
    )

    total_rows = 0
    live_keys = 0
    dead_keys = 0

    for source in [
        "source2",
        "source3"
    ]:

        print()
        print(
            f"Scanning {source}..."
        )

        path = os.path.join(
            DATASET_ROOT,
            split,
            f"{split}_{source}.tsv"
        )

        for chunk_no, chunk in enumerate(
            pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                usecols=[
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country"
                ],
                keep_default_na=False,
                chunksize=CHUNK
            )
        ):

            for row in chunk.itertuples(
                index=False
            ):

                target_id = row.entity_id

                country = (
                    str(
                        row.country
                    )
                    .strip()
                    .casefold()
                )

                name = normalize_name(
                    row.business_name
                )

                addr = normalize_address(
                    row.business_address
                )

                keys = target_keys(
                    country,
                    name,
                    addr
                )

                for signal_id, key_hash in keys:

                    pair = (
                        signal_id,
                        key_hash
                    )

                    if pair not in wanted:
                        continue

                    index[pair].append(
                        target_id
                    )

                    total_rows += 1


            if (
                chunk_no + 1
            ) % 4 == 0:

                print(
                    f"  {source} "
                    f"chunk={chunk_no+1} "
                    f"rows scanned={total_rows:,} "
                    f"keys={len(index):,}"
                )


    print()
    print(
        f"Raw live keys: {len(index):,}"
    )


    # ========================================================
    # CAP FREQUENT KEYS
    # ========================================================

    capped_index = {}

    for key, ids in index.items():

        signal_id, _ = key

        cap = CAPS[
            SIGNALS[signal_id]
        ]

        if len(ids) <= cap:

            # Deduplicate
            ids = list(
                dict.fromkeys(
                    ids
                )
            )

            capped_index[
                key
            ] = ids

            live_keys += 1

        else:

            dead_keys += 1


    print(
        f"Live keys after caps: "
        f"{live_keys:,}"
    )

    print(
        f"Dropped over-cap keys: "
        f"{dead_keys:,}"
    )

    print(
        f"Target postings: "
        f"{sum(len(v) for v in capped_index.values()):,}"
    )

    return capped_index


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    split,
    s1,
    s1_keys,
    index
):

    output_path = os.path.join(
        CACHE_DIR,
        f"{split}_candidate_pairs_v4.parquet"
    )

    if os.path.exists(
        output_path
    ):
        os.remove(
            output_path
        )


    print()
    print(
        "=" * 70
    )

    print(
        "GENERATING CANDIDATES"
    )

    print(
        "=" * 70
    )


    writer = None

    batch_s1 = []
    batch_target = []
    batch_signals = []

    total_pairs = 0

    candidate_counts = []

    t0 = time.time()


    def flush():

        nonlocal writer

        if not batch_s1:
            return


        table = pa.table({
            "s1_idx": pa.array(
                batch_s1,
                type=pa.int32()
            ),

            "cand_code": pa.array(
                batch_target,
                type=pa.int32()
            ),

            "n_signals": pa.array(
                batch_signals,
                type=pa.int16()
            )
        })


        if writer is None:

            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="zstd"
            )


        writer.write_table(
            table
        )


        batch_s1.clear()
        batch_target.clear()
        batch_signals.clear()


    # ========================================================
    # Target ID -> integer code
    # ========================================================

    all_target_ids = []

    print(
        "Creating target ID map..."
    )

    for source in [
        "source2",
        "source3"
    ]:

        path = os.path.join(
            DATASET_ROOT,
            split,
            f"{split}_{source}.tsv"
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id"
            ],
            chunksize=CHUNK
        ):

            all_target_ids.extend(
                chunk[
                    "entity_id"
                ].tolist()
            )


    target_to_code = {
        target_id: i
        for i, target_id
        in enumerate(
            all_target_ids
        )
    }


    # ========================================================
    # Candidate generation
    # ========================================================

    for s1_idx, (
        s1_id,
        keys
    ) in enumerate(
        s1_keys
    ):

        candidates = defaultdict(
            int
        )

        for signal_id, key_hash in keys:

            ids = index.get(
                (
                    signal_id,
                    key_hash
                )
            )

            if not ids:
                continue

            for target_id in ids:

                candidates[
                    target_id
                ] |= (
                    1 << signal_id
                )


        count = len(
            candidates
        )

        candidate_counts.append(
            count
        )


        for target_id, mask in candidates.items():

            batch_s1.append(
                s1_idx
            )

            batch_target.append(
                target_to_code[
                    target_id
                ]
            )

            batch_signals.append(
                mask.bit_count()
            )

            total_pairs += 1


        if (
            s1_idx + 1
        ) % 100_000 == 0:

            flush()

            print(
                f"  S1: "
                f"{s1_idx+1:,}/"
                f"{len(s1):,} | "
                f"pairs={total_pairs:,} | "
                f"time={time.time()-t0:.1f}s"
            )


    flush()

    if writer is not None:
        writer.close()


    print()
    print(
        f"Candidate pairs: "
        f"{total_pairs:,}"
    )

    print(
        f"S1 with candidates: "
        f"{sum(x > 0 for x in candidate_counts):,}/"
        f"{len(s1):,}"
    )

    if candidate_counts:

        s = pd.Series(
            candidate_counts
        )

        print(
            f"Mean: {s.mean():.2f}"
        )

        print(
            f"Median: {s.median():.0f}"
        )

        print(
            f"P95: {s.quantile(.95):.0f}"
        )

        print(
            f"P99: {s.quantile(.99):.0f}"
        )

        print(
            f"Max: {s.max():,}"
        )


    return (
        output_path,
        target_to_code
    )


# ============================================================
# GROUND TRUTH RECALL
# ============================================================

def evaluate_recall(
    split,
    candidate_path,
    s1,
    target_to_code
):

    if split != "train":
        return


    print()
    print(
        "=" * 70
    )

    print(
        "FULL GROUND-TRUTH RECALL"
    )

    print(
        "=" * 70
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


    # --------------------------------------------------------
    # Build S1 entity -> index
    # --------------------------------------------------------

    s1_to_idx = {
        entity_id: i
        for i, entity_id
        in enumerate(
            s1.entity_id
        )
    }


    # --------------------------------------------------------
    # Full true pair set
    #
    # Only true pairs whose target exists in our target map
    # are used for "recoverable" recall.
    # But we ALSO report targets not indexed.
    # --------------------------------------------------------

    truth = set()

    total_truth = 0
    not_indexed = 0


    for row in gt.itertuples(
        index=False
    ):

        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if (
            not matched
            or pd.isna(matched)
        ):
            continue


        s1_idx = s1_to_idx.get(
            s1_id
        )

        if s1_idx is None:
            continue


        for target_id in str(
            matched
        ).split(","):

            target_id = target_id.strip()

            if not target_id:
                continue


            total_truth += 1


            code = target_to_code.get(
                target_id
            )

            if code is None:

                not_indexed += 1

                continue


            truth.add(
                (
                    s1_idx,
                    code
                )
            )


    print(
        f"TRUE PAIRS: "
        f"{total_truth:,}"
    )

    print(
        f"Targets not in index: "
        f"{not_indexed:,}"
    )

    # --------------------------------------------------------
    # Stream candidates
    # --------------------------------------------------------

    recovered = set()

    pf = pq.ParquetFile(
        candidate_path
    )

    for batch in pf.iter_batches(
        columns=[
            "s1_idx",
            "cand_code"
        ],
        batch_size=1_000_000
    ):

        s1_values = batch[
            "s1_idx"
        ].to_numpy()

        target_values = batch[
            "cand_code"
        ].to_numpy()


        for a, b in zip(
            s1_values,
            target_values
        ):

            pair = (
                int(a),
                int(b)
            )

            if pair in truth:

                recovered.add(
                    pair
                )


    recovered_count = len(
        recovered
    )

    indexed_truth = len(
        truth
    )

    indexed_missed = (
        indexed_truth
        - recovered_count
    )

    print()
    print(
        f"RECOVERED: "
        f"{recovered_count:,}"
    )

    print(
        f"INDEXED TRUE PAIRS: "
        f"{indexed_truth:,}"
    )

    print(
        f"INDEXED MISSED: "
        f"{indexed_missed:,}"
    )

    print(
        f"TOTAL MISSED: "
        f"{total_truth - recovered_count:,}"
    )


    if total_truth:

        recall = (
            recovered_count
            / total_truth
        )

        print()
        print(
            f"FULL RECALL: "
            f"{recall:.4%}"
        )


    if indexed_truth:

        indexed_recall = (
            recovered_count
            / indexed_truth
        )

        print(
            f"RECALL AMONG INDEXED: "
            f"{indexed_recall:.4%}"
        )


# ============================================================
# SAVE MAPS
# ============================================================

def save_maps(
    split,
    s1,
    target_to_code
):

    path = os.path.join(
        CACHE_DIR,
        f"{split}_id_maps_v4.pkl"
    )

    data = {
        "s1_ids": s1[
            "entity_id"
        ].tolist(),

        "target_ids": [
            None
        ] * len(
            target_to_code
        )
    }


    for entity_id, code in target_to_code.items():

        data[
            "target_ids"
        ][code] = entity_id


    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


    print()
    print(
        f"Saved ID maps: {path}"
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

    split = args.split


    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )


    print(
        "=" * 70
    )

    print(
        "V4 ENTITY-RESOLUTION BLOCKER"
    )

    print(
        "=" * 70
    )

    print(
        f"Split: {split}"
    )


    # ========================================================
    # LOAD S1
    # ========================================================

    print()
    print(
        "Loading Source 1..."
    )

    s1 = load_source(
        split,
        "source1"
    )

    s1 = prepare_dataframe(
        s1
    )

    print(
        f"S1 rows: "
        f"{len(s1):,}"
    )


    # ========================================================
    # BUILD STATISTICS
    # ========================================================

    name_df = build_name_ngram_df(
        s1
    )

    address_df = build_address_token_df(
        s1
    )


    # ========================================================
    # S1 KEYS
    # ========================================================

    s1_keys, wanted = build_s1_keys(
        s1,
        name_df,
        address_df
    )


    # ========================================================
    # TARGET INDEX
    # ========================================================

    index = build_index(
        split,
        wanted
    )


    # ========================================================
    # CANDIDATES
    # ========================================================

    candidate_path, target_to_code = generate_candidates(
        split,
        s1,
        s1_keys,
        index
    )


    # ========================================================
    # MAPS
    # ========================================================

    save_maps(
        split,
        s1,
        target_to_code
    )


    # ========================================================
    # RECALL
    # ========================================================

    evaluate_recall(
        split,
        candidate_path,
        s1,
        target_to_code
    )


    print()
    print(
        "=" * 70
    )

    print(
        "V4 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Candidate file:"
        f"\n{candidate_path}"
    )


if __name__ == "__main__":
    main()