"""Train the PSD audio classifier from the cleaned class feature CSV.

Run from the project root:
    python train_model.py

The supplied 200-row feature CSV is deduplicated using the SHA-256 audit created
from the two raw class ZIPs. The cleaned modelling table has 176 independent
audio contents (88 per class). CV groups are parsed student/source IDs so that
samples from one recorder do not cross train/test folds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "features_model.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FEATURES = [
    "rms_mean", "centroid_mean", "zcr_mean",
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
]
CLASSES = ["non_berisik", "berisik"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    required = {"filename", "source_student_id", "target", *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")
    if df[FEATURES].isna().any().any():
        raise ValueError("Ada nilai kosong pada kolom fitur")
    if len(FEATURES) != 16:
        raise RuntimeError("Jumlah fitur seharusnya 16")

    X = df[FEATURES].astype(float)
    y = df["target"].astype(str)
    groups = df["source_student_id"].astype(str)

    if set(y.unique()) != set(CLASSES):
        raise ValueError(f"Label harus {CLASSES}, ditemukan {sorted(y.unique())}")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.empty(len(df), dtype=object)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = model.predict(X.iloc[test_idx])
        oof[test_idx] = prediction

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Group leakage pada fold {fold}: {sorted(overlap)}")

        fold_rows.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": len(overlap),
            "accuracy": accuracy_score(y.iloc[test_idx], prediction),
            "macro_f1": f1_score(y.iloc[test_idx], prediction, average="macro"),
        })

    metrics = {
        "accuracy": float(accuracy_score(y, oof)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "precision_macro": float(precision_score(y, oof, average="macro")),
        "recall_macro": float(recall_score(y, oof, average="macro")),
        "f1_macro": float(f1_score(y, oof, average="macro")),
    }
    cm = confusion_matrix(y, oof, labels=CLASSES).tolist()
    report = classification_report(
        y, oof, labels=CLASSES, target_names=["Non-berisik", "Berisik"],
        output_dict=True, zero_division=0,
    )

    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "feature_names": FEATURES},
                MODEL_DIR / "random_forest.joblib")

    metadata = {
        "model_name": "RandomForestClassifier",
        "classes": CLASSES,
        "feature_names": FEATURES,
        "feature_count": len(FEATURES),
        "feature_statistics": "mean_only",
        "sample_rate_hz": 22050,
        "audio_channels": "mono",
        "n_samples": len(df),
        "class_counts": y.value_counts().to_dict(),
        "source_group_count": int(groups.nunique()),
        "model_parameters": {
            "n_estimators": 200,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "evaluation": {
            "method": "5-fold StratifiedGroupKFold out-of-fold; groups=source_student_id",
            "metrics": metrics,
            "confusion_matrix_labels": CLASSES,
            "confusion_matrix": cm,
            "classification_report": report,
            "folds": fold_rows,
            "group_overlap": 0,
            "interpretation": (
                "Evaluasi mengukur kesesuaian terhadap label yang diberikan. "
                "Belum semua label divalidasi secara manual."
            ),
        },
        "training_note": "Model final di-fit ulang pada seluruh 176 sampel bersih setelah OOF CV.",
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "evaluation_results.json").write_text(
        json.dumps(metadata["evaluation"], indent=2), encoding="utf-8"
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    pred_df = df[["filename", "source_student_id", "target"]].copy()
    pred_df["oof_prediction"] = oof
    pred_df["correct"] = pred_df["target"] == pred_df["oof_prediction"]
    pred_df.to_csv(REPORT_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(REPORT_DIR / "grouped_cv_folds.csv", index=False)

    print("Model saved:", MODEL_DIR / "random_forest.joblib")
    print("Samples:", len(df), "| source groups:", groups.nunique())
    print("OOF metrics:", json.dumps(metrics, indent=2))
    print("Confusion matrix [non_berisik, berisik]:", cm)


if __name__ == "__main__":
    main()
