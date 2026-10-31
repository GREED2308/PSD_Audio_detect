"""Prepare the clean 16-feature dataset from the class CSV and SHA-256 manifest.

Run:
    python prepare_dataset.py

Same-content files inside one class are collapsed to one representative.
If identical audio appears under both labels, all rows for that hash are
excluded until the source label can be confirmed.
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "dataset_fitur_suara_original.csv"
MANIFEST = ROOT / "data" / "dataset_manifest.csv"
OUTPUT = ROOT / "data" / "features_model.csv"

FEATURES = [
    "rms_mean", "centroid_mean", "zcr_mean",
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
]


def main() -> None:
    data = pd.read_csv(RAW_CSV)
    manifest = pd.read_csv(MANIFEST)
    if data["nama_file"].duplicated().any():
        raise ValueError("Ada nama file ganda dalam CSV fitur.")
    joined = data.merge(
        manifest[["filename", "class_folder", "sha256", "student_id", "cross_label_duplicate"]],
        left_on="nama_file", right_on="filename", how="left", validate="one_to_one",
    )
    if joined["sha256"].isna().any():
        missing = joined.loc[joined.sha256.isna(), "nama_file"].tolist()
        raise ValueError(f"Manifest tidak memiliki hash untuk: {missing[:5]}")
    if joined[FEATURES].isna().any().any():
        raise ValueError("Ada nilai kosong pada fitur.")

    # Exclude every record whose identical content is assigned to both classes.
    conflict_hashes = set(
        joined.groupby("sha256")["class_folder"].nunique().loc[lambda s: s > 1].index
    )
    joined = joined[~joined["sha256"].isin(conflict_hashes)].copy()

    # Collapse duplicate content within the same class.
    joined = joined.drop_duplicates(subset="sha256", keep="first").copy()
    joined["target"] = joined["label"].map({0: "non_berisik", 1: "berisik"})
    joined["source_student_id"] = joined["student_id"].astype(str)

    result = joined[
        ["nama_file", "source_student_id", "sha256", "target", *FEATURES]
    ].rename(columns={"nama_file": "filename"})
    result.to_csv(OUTPUT, index=False)

    counts = result.target.value_counts().to_dict()
    if len(result) != 176 or counts != {"non_berisik": 88, "berisik": 88}:
        raise RuntimeError(
            f"Jumlah hasil pembersihan tidak sesuai ekspektasi: {len(result)}, {counts}"
        )
    print(f"Dataset bersih disimpan: {OUTPUT}")
    print(f"Jumlah sampel: {len(result)} | distribusi kelas: {counts}")
    print(f"Hash konflik yang dikeluarkan: {len(conflict_hashes)}")


if __name__ == "__main__":
    main()
