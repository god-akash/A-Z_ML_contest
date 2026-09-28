import os
import pandas as pd


TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"


def inspect_file(path):
    print("\n" + "=" * 70)
    print(f"FILE: {path}")
    print("=" * 70)

    df = pd.read_csv(path, sep="\t")

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    return df


# ============================================================
# TRAIN
# ============================================================

print("\n========== TRAIN DATA ==========")

train_s1 = inspect_file(
    os.path.join(TRAIN_DIR, "train_source1.tsv")
)

train_s2 = inspect_file(
    os.path.join(TRAIN_DIR, "train_source2.tsv")
)

train_s3 = inspect_file(
    os.path.join(TRAIN_DIR, "train_source3.tsv")
)

ground_truth = inspect_file(
    os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
)


# ============================================================
# TEST
# ============================================================

print("\n========== TEST DATA ==========")

test_s1 = inspect_file(
    os.path.join(TEST_DIR, "test_source1.tsv")
)

test_s2 = inspect_file(
    os.path.join(TEST_DIR, "test_source2.tsv")
)

test_s3 = inspect_file(
    os.path.join(TEST_DIR, "test_source3.tsv")
)


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

print("\n========== COUNTRY DISTRIBUTION ==========")

datasets = {
    "train_s1": train_s1,
    "train_s2": train_s2,
    "train_s3": train_s3,
    "test_s1": test_s1,
    "test_s2": test_s2,
    "test_s3": test_s3,
}

for name, df in datasets.items():

    print(f"\n{name}")

    if "country" in df.columns:
        print(
            df["country"]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# GROUND TRUTH
# ============================================================

print("\n========== GROUND TRUTH ==========")

print(ground_truth.head(10).to_string(index=False))

matches = ground_truth["matched_entity_ids"].fillna("")

match_counts = matches.apply(
    lambda x: 0 if str(x).strip() == ""
    else len(
        [
            item
            for item in str(x).split(",")
            if item.strip()
        ]
    )
)

print("\nMatch count distribution:")
print(match_counts.value_counts().sort_index())

print("\nTotal S1 entities:", len(ground_truth))
print("Singletons:", (match_counts == 0).sum())
print("With matches:", (match_counts > 0).sum())
print("Maximum matches:", match_counts.max())


# ============================================================
# DUPLICATE IDs
# ============================================================

print("\n========== DUPLICATE CHECK ==========")

for name, df in datasets.items():

    if "entity_id" in df.columns:

        duplicates = df["entity_id"].duplicated().sum()

        print(f"{name}: {duplicates} duplicate IDs")


print("\nInspection complete.")