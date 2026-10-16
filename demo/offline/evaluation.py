
"""Offline evaluation and error inspection for Experiment 7F."""

import argparse

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from demo.inference.predictor import (
    PCA_DIR,
    predict_split,
)

console = Console()


def normalize_label(value):
    """Convert supported dataset labels to integer 0 or 1."""
    if pd.isna(value):
        raise ValueError("A commit has a missing reference label.")

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"1", "true", "buggy", "bug"}:
            return 1

        if normalized in {
            "0", "false", "non-buggy", "non_buggy", "nonbuggy"
        }:
            return 0

        raise ValueError(f"Unrecognized reference label: {value!r}")

    if value in (0, 1, False, True):
        return int(value)

    raise ValueError(f"Unrecognized reference label: {value!r}")


def load_evaluation_data(split):
    """Combine model predictions with the recorded dataset labels."""
    predictions = predict_split(split)

    metadata = pd.read_csv(
        PCA_DIR / f"{split}_pca384.csv",
        usecols=lambda column: column in {
            "commit_id", "buggy", "project", "message",
            "language", "year",
        },
    )

    predictions["commit_id"] = predictions["commit_id"].astype(str)
    metadata["commit_id"] = metadata["commit_id"].astype(str)

    if predictions["commit_id"].duplicated().any():
        raise ValueError("Duplicate commit IDs in predictions.")

    if metadata["commit_id"].duplicated().any():
        raise ValueError("Duplicate commit IDs in metadata.")

    data = predictions.merge(
        metadata,
        on="commit_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not data["_merge"].eq("both").all():
        raise ValueError("Some predictions lack matching dataset metadata.")

    data.drop(columns="_merge", inplace=True)
    data["actual"] = data["buggy"].map(normalize_label)
    data["predicted"] = data["blend_prediction"].astype(int)

    data["error_type"] = "Correct"
    data.loc[
        (data["actual"] == 0) & (data["predicted"] == 1),
        "error_type",
    ] = "False Positive"
    data.loc[
        (data["actual"] == 1) & (data["predicted"] == 0),
        "error_type",
    ] = "False Negative"
    data.loc[
        (data["actual"] == 1) & (data["predicted"] == 1),
        "error_type",
    ] = "True Positive"
    data.loc[
        (data["actual"] == 0) & (data["predicted"] == 0),
        "error_type",
    ] = "True Negative"

    return data


def show_summary(data):
    """Display the confusion matrix and evaluation metrics."""
    actual = data["actual"]
    predicted = data["predicted"]
    probabilities = data["blend_probability"]

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    summary = Table(
        title="Confusion Matrix",
        box=box.ROUNDED,
    )
    summary.add_column("Actual \\ Predicted")
    summary.add_column("Non-buggy (0)", justify="right")
    summary.add_column("Buggy (1)", justify="right")

    summary.add_row("Non-buggy (0)", f"{tn:,}", f"{fp:,}")
    summary.add_row("Buggy (1)", f"{fn:,}", f"{tp:,}")
    console.print(summary)

    metrics = Table(title="Test Evaluation Metrics", box=box.SIMPLE)
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")

    metric_values = [
        ("Accuracy", accuracy_score(actual, predicted)),
        ("Precision", precision_score(actual, predicted, zero_division=0)),
        ("Recall", recall_score(actual, predicted, zero_division=0)),
        ("F1-score", f1_score(actual, predicted, zero_division=0)),
        ("MCC", matthews_corrcoef(actual, predicted)),
        ("ROC-AUC", roc_auc_score(actual, probabilities)),
        ("PR-AUC (Average Precision)", average_precision_score(
            actual, probabilities
        )),
    ]

    for name, value in metric_values:
        metrics.add_row(name, f"{value:.4f}")

    console.print(metrics)

    error_table = Table(title="Prediction Outcome Counts", box=box.SIMPLE)
    error_table.add_column("Outcome")
    error_table.add_column("Count", justify="right")

    for outcome in [
        "True Positive", "True Negative",
        "False Positive", "False Negative",
    ]:
        count = int((data["error_type"] == outcome).sum())
        error_table.add_row(outcome, f"{count:,}")

    console.print(error_table)


def show_error_list(data, error_type):
    """Browse errors and inspect individual commits."""
    errors = data.loc[data["error_type"] == error_type].copy()
    errors = errors.sort_values(
        "blend_probability", ascending=False
    ).reset_index(drop=True)

    if errors.empty:
        console.print(f"[green]No {error_type.lower()} cases found.[/green]")
        return

    console.print(
        Panel(
            f"[bold]{len(errors):,} {error_type} cases[/bold]\n"
            + (
                "The model predicted buggy, but the recorded label is non-buggy."
                if error_type == "False Positive"
                else
                "The model predicted non-buggy, but the recorded label is buggy."
            ),
            title="Error Inspection",
            border_style="yellow",
        )
    )

    page_size = 20
    start = 0

    while start < len(errors):
        page = errors.iloc[start:start + page_size]

        table = Table(
            title=f"Cases {start + 1}–{start + len(page)} "
                  f"of {len(errors)}",
            box=box.SIMPLE,
        )
        table.add_column("#", justify="right")
        table.add_column("Commit ID")
        table.add_column("Project")
        table.add_column("Blend probability", justify="right")
        table.add_column("Commit message", max_width=48)

        for index, row in page.iterrows():
            message = str(row.get("message", ""))
            if message == "nan":
                message = ""

            project = str(row.get("project", "Unknown"))
            if project == "nan":
                project = "Unknown"

            table.add_row(
                str(index + 1),
                str(row["commit_id"]),
                project,
                f"{row['blend_probability']:.2%}",
                message.replace("\n", " ")[:100],
            )

        console.print(table)

        choice = Prompt.ask(
            "Enter a case number to inspect, 'n' for next page, "
            "'p' for previous page, or 'q' to return"
        ).strip().lower()

        if choice == "q":
            return

        if choice == "n":
            if start + page_size < len(errors):
                start += page_size
            else:
                console.print("[yellow]Already at the last page.[/yellow]")
            continue

        if choice == "p":
            start = max(0, start - page_size)
            continue

        try:
            selected = int(choice)
        except ValueError:
            console.print("[red]Invalid choice.[/red]")
            continue

        if selected < 1 or selected > len(errors):
            console.print("[red]Case number out of range.[/red]")
            continue

        row = errors.iloc[selected - 1]

        details = Table(box=box.SIMPLE)
        details.add_column("Field")
        details.add_column("Value")

        details.add_row("Commit ID", str(row["commit_id"]))
        details.add_row("Project", str(row.get("project", "Unknown")))
        details.add_row("Error type", str(row["error_type"]))
        details.add_row("Actual label", str(row["actual"]))
        details.add_row("Predicted label", str(row["predicted"]))
        details.add_row(
            "Blended probability",
            f"{row['blend_probability']:.4f}",
        )
        details.add_row(
            "Random Forest",
            f"{row['rf_probability']:.4f}",
        )
        details.add_row("XGBoost", f"{row['xgb_probability']:.4f}")
        details.add_row(
            "LightGBM",
            f"{row['lightgbm_probability']:.4f}",
        )
        details.add_row(
            "Commit message",
            str(row.get("message", ""))[:1500],
        )

        console.print(details)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment 7F and inspect prediction errors."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]Experiment 7F — Offline Evaluation[/bold cyan]\n"
            f"Split: {args.split} | Threshold: 0.50",
            border_style="cyan",
        )
    )
    console.print("Loading predictions and reference labels...")

    data = load_evaluation_data(args.split)
    show_summary(data)

    while True:
        console.print("\n[bold]Error inspection menu[/bold]")
        console.print("1. Browse false positives")
        console.print("2. Browse false negatives")
        console.print("3. Browse all incorrect predictions")
        console.print("4. Show evaluation summary again")
        console.print("q. Quit")

        choice = Prompt.ask("Choose an option").strip().lower()

        if choice == "1":
            show_error_list(data, "False Positive")
        elif choice == "2":
            show_error_list(data, "False Negative")
        elif choice == "3":
            errors = data.loc[
                data["error_type"].isin(["False Positive", "False Negative"])
            ].copy()

            console.print(
                f"Incorrect predictions: {len(errors):,}"
            )
            if errors.empty:
                console.print("[green]No errors found.[/green]")
                continue

            error_choice = Prompt.ask(
                "Browse which type? (fp/fn)"
            ).strip().lower()

            if error_choice == "fp":
                show_error_list(data, "False Positive")
            elif error_choice == "fn":
                show_error_list(data, "False Negative")
            else:
                console.print("[yellow]Choose 'fp' or 'fn'.[/yellow]")
        elif choice == "4":
            show_summary(data)
        elif choice in {"q", "quit", "exit"}:
            console.print("Evaluation finished.")
            break
        else:
            console.print("[red]Choose 1, 2, 3, 4, or q.[/red]")


if __name__ == "__main__":
    main()
