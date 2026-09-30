# src/train_xgboost.py
import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)

ROOT = r"D:\A-Z_ML_contest"

FEATURE_FILE = os.path.join(ROOT, "cache", "training_features.parquet")
MODEL_FILE = os.path.join(ROOT, "cache", "entity_match_xgb.json")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRAINING XGBOOST ENTITY MATCHING MODEL")
print("=" * 70)

print("\nLoading features...")
df = pd.read_parquet(FEATURE_FILE)
print("Rows:", len(df))


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "name_ratio",
    "name_token_set",
    "name_partial",
    "name_exact",
    "name_token_jaccard",
    "name_len_diff",

    "address_ratio",
    "address_token_set",
    "address_partial",
    "address_exact",
    "address_token_jaccard",
    "address_len_diff",

    "number_overlap",
    "number_count_common",

    "country_match",
    "n_signals",

    "name_missing",
    "address_missing",
]

X_train = df.loc[df["split"] == "train", FEATURES].copy()
y_train = df.loc[df["split"] == "train", "label"]

X_valid = df.loc[df["split"] == "valid", FEATURES].copy()
y_valid = df.loc[df["split"] == "valid", "label"]

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_valid))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nValidation labels:")
print(y_valid.value_counts())


# ============================================================
# CLASS BALANCE
# ============================================================

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / n_pos

print(f"\nClass ratio (neg/pos) in training set: {scale_pos_weight:.3f}")
print("Using this as scale_pos_weight.")


# ============================================================
# MODEL
# ============================================================

print("\nCreating XGBoost model...")

model = XGBClassifier(
    n_estimators=700,
    max_depth=7,
    learning_rate=0.07,

    subsample=0.85,
    colsample_bytree=0.85,

    min_child_weight=3,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",
    eval_metric="logloss",

    tree_method="hist",

    early_stopping_rounds=30,

    n_jobs=-1,
    random_state=42,
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining...")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_valid, y_valid),
    ],
    verbose=50,
)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Best validation logloss: {model.best_score:.5f}")


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

valid_prob = model.predict_proba(X_valid)[:, 1]

print("\nValidation probability distribution:")
print(pd.Series(valid_prob).describe())


# ============================================================
# FIND BEST F0.5 THRESHOLD
# ============================================================

print("\nSearching probability threshold...")

results = []

for threshold in np.arange(0.10, 0.96, 0.01):
    pred = (valid_prob >= threshold).astype(int)

    precision = precision_score(y_valid, pred, zero_division=0)
    recall = recall_score(y_valid, pred, zero_division=0)
    f05 = fbeta_score(y_valid, pred, beta=0.5, zero_division=0)

    results.append((threshold, precision, recall, f05))

results = pd.DataFrame(
    results, columns=["threshold", "precision", "recall", "f0.5"]
)

best = results.loc[results["f0.5"].idxmax()]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS (top 15 by F0.5)")
print("=" * 70)

print(results.sort_values("f0.5", ascending=False).head(15).to_string(index=False))

print("\nFull threshold sweep (every 0.05, for context):")
print(results[results["threshold"].round(2).isin(np.arange(0.10, 0.96, 0.05).round(2))]
     .to_string(index=False))

print("\nSelected validation threshold:")
print(f"threshold = {best['threshold']:.2f}")
print(f"precision = {best['precision']:.4f}")
print(f"recall    = {best['recall']:.4f}")
print(f"F0.5      = {best['f0.5']:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

best_pred = (valid_prob >= best["threshold"]).astype(int)

print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_valid, best_pred))


# ============================================================
# CONSERVATIVE THRESHOLD CHECK
# ============================================================

# Since singletons score 0 on ANY false match, also show a higher, more
# conservative threshold explicitly -- useful to compare against "best" when
# deciding the final production cutoff.
for t in [0.5, 0.7, 0.8, 0.9]:
    pred = (valid_prob >= t).astype(int)
    p = precision_score(y_valid, pred, zero_division=0)
    r = recall_score(y_valid, pred, zero_division=0)
    f = fbeta_score(y_valid, pred, beta=0.5, zero_division=0)
    print(f"  threshold={t:.2f}  precision={p:.4f}  recall={r:.4f}  F0.5={f:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
print(importance.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(MODEL_FILE)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)
print("Model:", MODEL_FILE)
print("Chosen threshold for later inference:", f"{best['threshold']:.2f}")