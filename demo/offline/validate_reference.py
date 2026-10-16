
"""Compare demo inference against frozen Experiment 7F predictions."""

import argparse

import numpy as np
import pandas as pd

from demo.inference.predictor import predict_split, EXPERIMENT_DIR


PROBABILITY_COLUMNS = [
    "rf_probability",
    "xgb_probability",
    "lightgbm_probability",
    "blend_probability",
]


def main():
    parser = argparse.ArgumentParser(
        description="Validate demo predictions against Experiment 7F."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Canonical split to validate (default: test).",
    )
    args = parser.parse_args()

    reference_path = (
        EXPERIMENT_DIR / f"{args.split}_predictions.csv"
    )

    print(f"Split: {args.split}")
    print(f"Reference: {reference_path}")
    print("Running inference with the saved models...")

    actual = predict_split(args.split)
    reference = pd.read_csv(reference_path)

    if actual["commit_id"].duplicated().any():
        raise ValueError("Duplicate commit IDs in demo predictions.")

    if reference["commit_id"].duplicated().any():
        raise ValueError("Duplicate commit IDs in reference predictions.")

    actual["commit_id"] = actual["commit_id"].astype(str)
    reference["commit_id"] = reference["commit_id"].astype(str)

    merged = reference.merge(
        actual,
        on="commit_id",
        how="outer",
        suffixes=("_reference", "_actual"),
        indicator=True,
        validate="one_to_one",
    )

    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        print(f"FAIL: {int(unmatched.sum())} unmatched commit IDs.")
        print(merged.loc[unmatched, ["commit_id", "_merge"]].head(10))
        raise SystemExit(1)

    print(f"Matched commits: {len(merged):,}")

    all_close = True

    for column in PROBABILITY_COLUMNS:
        expected = merged[f"{column}_reference"].to_numpy(dtype=float)
        observed = merged[f"{column}_actual"].to_numpy(dtype=float)

        differences = np.abs(expected - observed)
        max_difference = float(differences.max())

        close = np.allclose(
            expected,
            observed,
            rtol=1e-5,
            atol=1e-6,
        )

        print(
            f"{column}: max_abs_diff={max_difference:.10g}; "
            f"within_tolerance={close}"
        )

        if not close:
            all_close = False

    expected_labels = merged["blend_prediction_reference"].to_numpy()
    observed_labels = merged["blend_prediction_actual"].to_numpy()

    label_matches = int(np.sum(expected_labels == observed_labels))
    print(
        f"Matching blend predictions: {label_matches:,}/{len(merged):,}"
    )

    if all_close and label_matches == len(merged):
        print("PASS: demo predictions match the reference within tolerance.")
    else:
        print("FAIL: predictions differ from the frozen reference.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
