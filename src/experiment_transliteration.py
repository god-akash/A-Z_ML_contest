import os
import re
import unicodedata
import pandas as pd

from unidecode import unidecode
from rapidfuzz.fuzz import ratio, token_set_ratio


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset", "train")


S1_PATH = os.path.join(DATA, "train_source1.tsv")
S2_PATH = os.path.join(DATA, "train_source2.tsv")
S3_PATH = os.path.join(DATA, "train_source3.tsv")
GT_PATH = os.path.join(DATA, "train_ground_truth.tsv")


def normalize(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transliterate(text):
    if not text:
        return ""

    text = unidecode(str(text))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_suffixes(text):
    suffixes = {
        "pvt", "ltd", "private", "limited",
        "llc", "inc", "corp", "corporation",
        "co", "company", "gmbh", "llp", "plc", "lp"
    }

    return " ".join(
        x for x in text.split()
        if x not in suffixes
    )


def name(text):
    return remove_suffixes(normalize(text))


def tname(text):
    return remove_suffixes(transliterate(text))


print("=" * 70)
print("FAST TRANSLITERATION EXPERIMENT")
print("=" * 70)

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "country"],
    keep_default_na=False
)

print("S1 loaded:", len(s1))


print("\nLoading S2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "country"],
    keep_default_na=False
)

print("S2 loaded:", len(s2))


print("\nLoading S3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "country"],
    keep_default_na=False
)

print("S3 loaded:", len(s3))


# ------------------------------------------------------------
# Combine sources
# ------------------------------------------------------------

other = pd.concat(
    [s2, s3],
    ignore_index=True
)

del s2
del s3


print("\nCreating normalized columns...")

other["normal_name"] = other["business_name"].map(name)

other["trans_name"] = other["business_name"].map(tname)

other["country_norm"] = (
    other["country"]
    .astype(str)
    .str.casefold()
)


# ------------------------------------------------------------
# Build only transliteration index
# ------------------------------------------------------------

print("\nBuilding transliteration index...")

trans_index = {}

for row in other[
    ["entity_id", "trans_name", "country_norm"]
].itertuples(index=False):

    if not row.trans_name:
        continue

    key = (
        row.country_norm,
        row.trans_name
    )

    if key not in trans_index:
        trans_index[key] = []

    trans_index[key].append(row.entity_id)


print(
    "Transliteration keys:",
    f"{len(trans_index):,}"
)


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


# ------------------------------------------------------------
# S1 lookup
# ------------------------------------------------------------

s1_lookup = {}

for row in s1[
    ["entity_id", "business_name", "country"]
].itertuples(index=False):

    s1_lookup[row.entity_id] = (
        row.business_name,
        str(row.country).casefold()
    )


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

true_pairs = 0
exact_normal_recovered = 0
trans_recovered = 0
new_recovered = 0

examples = []


print("\nEvaluating ground truth...")


for row in gt.itertuples(index=False):

    sid = row.source1_entity_id
    matches = str(row.matched_entity_ids).strip()

    if not matches:
        continue

    if sid not in s1_lookup:
        continue

    s1_name, country = s1_lookup[sid]

    normal = name(s1_name)
    trans = tname(s1_name)

    normal_key = (country, normal)
    trans_key = (country, trans)

    normal_candidates = set()

    if normal:
        normal_candidates = {
            x
            for x in trans_index.get(normal_key, [])
        }

    trans_candidates = set(
        trans_index.get(trans_key, [])
    )

    for target in matches.split(","):

        target = target.strip()

        if not target:
            continue

        true_pairs += 1

        if target in normal_candidates:
            exact_normal_recovered += 1

        if target in trans_candidates:
            trans_recovered += 1

            if target not in normal_candidates:
                new_recovered += 1

                if len(examples) < 30:

                    target_row = other[
                        other["entity_id"] == target
                    ]

                    if len(target_row):

                        examples.append(
                            (
                                sid,
                                target,
                                s1_name,
                                target_row.iloc[0]["business_name"]
                            )
                        )


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"TRUE PAIRS:                 {true_pairs:,}")
print(
    f"NORMAL EXACT RECOVERED:     "
    f"{exact_normal_recovered:,}"
)

print(
    f"TRANSLITERATION RECOVERED:  "
    f"{trans_recovered:,}"
)

print(
    f"NEW PAIRS FROM TRANSLIT:    "
    f"{new_recovered:,}"
)

if true_pairs:

    print(
        f"\nTRANSLIT RECALL:            "
        f"{trans_recovered / true_pairs:.4%}"
    )

    print(
        f"NEW RECOVERY RATE:          "
        f"{new_recovered / true_pairs:.4%}"
    )


print("\nExamples of NEW transliteration matches:")
print("-" * 70)

for sid, tid, n1, n2 in examples:

    print("\nS1:", sid)
    print("OTHER:", tid)
    print("NAME 1:", n1)
    print("NAME 2:", n2)


print("\nDone.")