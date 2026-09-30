import os
import pickle
import time
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq


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


# ============================================================
# FILES CREATED BY build_candidates_v3.py
# ============================================================

INDEX_PATH = os.path.join(
    CACHE_DIR,
    "train_v3_index.pkl"
)

ID_MAP_PATH = os.path.join(
    CACHE_DIR,
    "train_v3_id_maps.pkl"
)

CANDIDATE_PATH = os.path.join(
    CACHE_DIR,
    "train_v3_candidate_pairs.parquet"
)

GT_PATH = os.path.join(
    DATASET_ROOT,
    "train",
    "train_ground_truth.tsv"
)


# ============================================================
# EVALUATE TRAIN RECALL
# ============================================================

def evaluate_train_recall(
    candidate_path,
    s1_ids,
    rid_ids
):

    start_time = time.time()

    print()
    print("=" * 70)
    print("V3 TRAIN BLOCKING RECALL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        f"Ground-truth rows: "
        f"{len(gt):,}"
    )

    # --------------------------------------------------------
    # S1 entity ID -> integer index
    # --------------------------------------------------------

    print("\nBuilding S1 ID map...")

    s1_to_idx = {
        eid: i
        for i, eid in enumerate(s1_ids)
    }

    print(
        f"S1 IDs: "
        f"{len(s1_to_idx):,}"
    )

    # --------------------------------------------------------
    # Target entity ID -> candidate integer code
    #
    # IMPORTANT:
    # Only records that appeared in the blocking index
    # have a code.
    # --------------------------------------------------------

    print("\nBuilding target ID map...")

    rid_to_code = {
        eid: i
        for i, eid in enumerate(rid_ids)
    }

    print(
        f"Indexed target records: "
        f"{len(rid_to_code):,}"
    )

    # --------------------------------------------------------
    # Build ground truth
    #
    # CRITICAL FIX:
    #
    # We count EVERY ground-truth pair.
    #
    # If a true target is not present in rid_to_code,
    # it is still counted in truth_total.
    #
    # That pair is simply considered unrecovered.
    # --------------------------------------------------------

    print("\nBuilding FULL ground truth...")

    truth_total = 0

    # Only pairs whose target exists in the blocking index
    # can possibly be recovered from the candidate parquet.
    truth_mapped = set()

    missing_target_count = 0

    s1_without_gt = 0

    for row in gt.itertuples(
        index=False
    ):

        # ----------------------------------------------------
        # Find S1 index
        # ----------------------------------------------------

        s1_idx = s1_to_idx.get(
            row.source1_entity_id
        )

        if s1_idx is None:
            continue

        # ----------------------------------------------------
        # Read matched target IDs
        # ----------------------------------------------------

        value = str(
            row.matched_entity_ids
        ).strip()

        # Empty = this S1 has no matches
        if not value:
            s1_without_gt += 1
            continue

        targets = [
            target.strip()
            for target in value.split(",")
            if target.strip()
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Count ALL targets, regardless of whether they
        # appear in the blocking index.
        # ----------------------------------------------------

        truth_total += len(
            targets
        )

        # ----------------------------------------------------
        # Create packed integer representation for targets
        # that ARE in the candidate index.
        # ----------------------------------------------------

        for target in targets:

            code = rid_to_code.get(
                target
            )

            if code is None:

                # Blocking completely missed this target.
                missing_target_count += 1

                continue

            packed_key = (
                s1_idx
                *
                (
                    len(rid_ids) + 1
                )
                +
                code
            )

            truth_mapped.add(
                packed_key
            )

    print()
    print(
        f"TRUE PAIRS (FULL GT): "
        f"{truth_total:,}"
    )

    print(
        f"TRUE TARGETS NOT IN INDEX: "
        f"{missing_target_count:,}"
    )

    print(
        f"TRUTH PAIRS AVAILABLE FOR "
        f"CANDIDATE CHECK: "
        f"{len(truth_mapped):,}"
    )

    # --------------------------------------------------------
    # Scan candidate parquet
    # --------------------------------------------------------

    print(
        "\nScanning candidate parquet..."
    )

    print(
        candidate_path
    )

    if not os.path.exists(
        candidate_path
    ):

        raise FileNotFoundError(
            "\nCandidate file not found:\n"
            + candidate_path
        )

    pf = pq.ParquetFile(
        candidate_path
    )

    recovered = 0

    # Number of candidates per S1
    per_s1 = defaultdict(int)

    processed_rows = 0

    batch_number = 0

    for batch in pf.iter_batches(
        batch_size=5_000_000
    ):

        batch_number += 1

        # ----------------------------------------------------
        # Convert columns to numpy
        # ----------------------------------------------------

        s1_arr = batch[
            "s1_idx"
        ].to_numpy()

        cand_arr = batch[
            "cand_code"
        ].to_numpy()

        processed_rows += len(
            s1_arr
        )

        # ----------------------------------------------------
        # Candidate count per S1
        # ----------------------------------------------------

        for x in s1_arr:

            per_s1[
                int(x)
            ] += 1

        # ----------------------------------------------------
        # Pack candidate pair:
        #
        #     s1_idx * (number_of_targets + 1)
        #     + target_code
        #
        # Same representation used for truth_mapped.
        # ----------------------------------------------------

        keys = (
            s1_arr.astype(
                "int64"
            )
            *
            (
                len(rid_ids) + 1
            )
            +
            cand_arr.astype(
                "int64"
            )
        )

        # ----------------------------------------------------
        # Check intersection with ground truth.
        #
        # Every candidate batch is checked against the set
        # of TRUE pairs that could be represented.
        # ----------------------------------------------------

        batch_keys = set(
            keys.tolist()
        )

        recovered_in_batch = len(
            batch_keys
            &
            truth_mapped
        )

        recovered += (
            recovered_in_batch
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"  batch={batch_number:,} "
            f"rows={processed_rows:,} "
            f"recovered={recovered:,}",
            flush=True
        )

    # --------------------------------------------------------
    # Calculate exact recall
    # --------------------------------------------------------

    if truth_total > 0:

        recall = (
            recovered
            /
            truth_total
        )

    else:

        recall = 0.0

    missed = (
        truth_total
        -
        recovered
    )

    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

    if per_s1:

        counts = pd.Series(
            list(
                per_s1.values()
            )
        )

        mean_candidates = (
            counts.mean()
        )

        median_candidates = (
            counts.median()
        )

        p95_candidates = (
            counts.quantile(
                0.95
            )
        )

        p99_candidates = (
            counts.quantile(
                0.99
            )
        )

        max_candidates = (
            counts.max()
        )

    else:

        mean_candidates = 0
        median_candidates = 0
        p95_candidates = 0
        p99_candidates = 0
        max_candidates = 0

    # --------------------------------------------------------
    # S1 with candidates
    # --------------------------------------------------------

    s1_with_candidates = len(
        per_s1
    )

    s1_total = len(
        s1_ids
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL V3 BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"TRUE PAIRS           : "
        f"{truth_total:,}"
    )

    print(
        f"RECOVERED            : "
        f"{recovered:,}"
    )

    print(
        f"MISSED               : "
        f"{missed:,}"
    )

    print(
        f"RECALL               : "
        f"{recall:.6f}"
    )

    print(
        f"RECALL (%)           : "
        f"{recall * 100:.4f}%"
    )

    print()

    print(
        f"GT TARGETS NOT INDEXED: "
        f"{missing_target_count:,}"
    )

    print()

    print(
        f"S1 WITH CANDIDATES   : "
        f"{s1_with_candidates:,}/"
        f"{s1_total:,}"
    )

    print()

    print(
        f"MEAN CANDIDATES      : "
        f"{mean_candidates:.1f}"
    )

    print(
        f"MEDIAN               : "
        f"{median_candidates:.0f}"
    )

    print(
        f"P95                  : "
        f"{p95_candidates:.0f}"
    )

    print(
        f"P99                  : "
        f"{p99_candidates:.0f}"
    )

    print(
        f"MAX                  : "
        f"{max_candidates:,.0f}"
    )

    print()

    print(
        f"Candidate rows read  : "
        f"{processed_rows:,}"
    )

    print(
        f"Evaluation time      : "
        f"{time.time() - start_time:.1f}s"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print("\nSANITY CHECKS")

    if truth_total == 7_638_365:

        print(
            "✓ Ground truth count is "
            "7,638,365"
        )

    else:

        print(
            "⚠ WARNING: Ground truth count "
            f"is {truth_total:,}, expected "
            "7,638,365"
        )

    if recovered <= truth_total:

        print(
            "✓ Recovered <= true pairs"
        )

    else:

        print(
            "⚠ WARNING: recovered > "
            "true pairs"
        )

    if 0 <= recall <= 1:

        print(
            "✓ Recall is within [0, 1]"
        )

    else:

        print(
            "⚠ WARNING: invalid recall"
        )

    print(
        "\nEvaluation complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXISTING V3 CANDIDATE EVALUATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        INDEX_PATH,
        ID_MAP_PATH,
        CANDIDATE_PATH,
        GT_PATH
    ]

    print("\nChecking required files...")

    for path in required_files:

        exists = os.path.exists(
            path
        )

        print(
            f"{'✓' if exists else '✗'} "
            f"{path}"
        )

        if not exists:

            raise FileNotFoundError(
                "\nRequired file missing:\n"
                + path
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

        id_maps = pickle.load(
            f
        )

    s1_ids = id_maps[
        "s1_ids"
    ]

    rid_ids = id_maps[
        "rid_ids"
    ]

    print(
        f"S1 records: "
        f"{len(s1_ids):,}"
    )

    print(
        f"Target records: "
        f"{len(rid_ids):,}"
    )

    # --------------------------------------------------------
    # Evaluate EXISTING parquet
    # --------------------------------------------------------

    evaluate_train_recall(
        CANDIDATE_PATH,
        s1_ids,
        rid_ids
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()