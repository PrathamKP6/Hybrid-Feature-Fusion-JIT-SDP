
"""Rich command-line interface for the Experiment 7F offline demo."""

import argparse
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from demo.inference.predictor import predict_split

console = Console()
ROOT = Path(__file__).resolve().parents[2]
PCA_DIR = ROOT / "results" / "pca"


def load_commit_metadata(split):
    """Load commit descriptions and labels for the selected split."""
    frame = pd.read_csv(PCA_DIR / f"{split}_pca384.csv")

    columns = [
        column
        for column in [
            "commit_id", "project", "year",
            "author_date", "message", "buggy",
        ]
        if column in frame.columns
    ]

    return frame[columns].copy()


def build_results(split):
    """Combine model predictions with commit metadata."""
    predictions = predict_split(split)
    metadata = load_commit_metadata(split)

    metadata["commit_id"] = metadata["commit_id"].astype(str)
    predictions["commit_id"] = predictions["commit_id"].astype(str)

    results = metadata.merge(
        predictions,
        on="commit_id",
        how="inner",
        validate="one_to_one",
    )

    if "project" in results.columns:
        results["project"] = results["project"].fillna("Unknown").astype(str)

    return results


def show_prediction(row, split):
    probability = float(row["blend_probability"])
    prediction = int(row["blend_prediction"])

    risk_label = (
        "Higher estimated defect risk"
        if prediction
        else "Lower estimated defect risk"
    )
    risk_style = "bold red" if prediction else "bold green"

    console.print()
    console.print(
        Panel(
            f"[bold]Commit:[/bold] {row['commit_id']}\n"
            f"[bold]Project:[/bold] {row.get('project', 'Unknown')}\n"
            f"[bold]Year:[/bold] {row.get('year', 'Unknown')}\n"
            f"[bold]Split:[/bold] {split}",
            title="Experiment 7F — Commit Analysis",
            border_style="cyan",
        )
    )

    console.print(
        Panel(
            f"[{risk_style}]{risk_label}[/{risk_style}]\n"
            f"Model-estimated defect probability: "
            f"[bold]{probability:.2%}[/bold]\n"
            f"Classification threshold: 50%",
            title="Blended Model Result",
            border_style="red" if prediction else "green",
        )
    )

    table = Table(
        title="Individual Model Outputs",
        box=box.ROUNDED,
        show_header=True,
    )
    table.add_column("Model")
    table.add_column("Estimated probability", justify="right")

    for name, column in [
        ("Random Forest", "rf_probability"),
        ("XGBoost", "xgb_probability"),
        ("LightGBM", "lightgbm_probability"),
    ]:
        table.add_row(name, f"{float(row[column]):.4f}")

    console.print(table)

    message = str(row.get("message", ""))
    if message and message.lower() != "nan":
        console.print(
            Panel(
                message[:1200],
                title="Commit Message",
                border_style="blue",
            )
        )

    if "buggy" in row.index and pd.notna(row["buggy"]):
        raw_label = row["buggy"]

        if isinstance(raw_label, str):
            normalized = raw_label.strip().lower()
            if normalized in {"true", "1", "buggy", "bug"}:
                label = "Buggy"
            elif normalized in {
                "false", "0", "non-buggy", "non_buggy", "nonbuggy"
            }:
                label = "Non-buggy"
            else:
                label = f"Unknown ({raw_label})"
        else:
            label = "Buggy" if bool(raw_label) else "Non-buggy"

        console.print(f"Reference label: {label}")

    console.print(
        "[dim]These are model estimates, not calibrated production "
        "probabilities or a guarantee that a commit contains a defect.[/dim]"
    )


def show_commit_list(results, title="Matching commits", limit=30):
    """Display a numbered preview of matching commits."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("No.", justify="right")
    table.add_column("Commit hash")
    table.add_column("Project")
    table.add_column("Risk estimate", justify="right")
    table.add_column("Prediction")

    preview = results.head(limit)

    for number, (_, row) in enumerate(preview.iterrows(), start=1):
        probability = float(row["blend_probability"])
        prediction = int(row["blend_prediction"])
        table.add_row(
            str(number),
            str(row["commit_id"])[:16],
            str(row.get("project", "Unknown"))[:30],
            f"{probability:.2%}",
            "Higher risk" if prediction else "Lower risk",
        )

    console.print(table)

    if len(results) > limit:
        console.print(
            f"[dim]Showing {limit} of {len(results):,} matches. "
            "Use search to narrow the results.[/dim]"
        )


def inspect_search_results(matches, split):
    """Let the user inspect one matching commit."""
    if matches.empty:
        console.print("[yellow]No matching commits found.[/yellow]")
        return

    show_commit_list(matches, title="Search results")

    while True:
        choice = Prompt.ask(
            "Enter a result number to inspect, or 'b' to go back"
        ).strip()

        if choice.lower() in {"b", "back", "q"}:
            return

        try:
            index = int(choice)
        except ValueError:
            console.print("[red]Enter a result number or 'b'.[/red]")
            continue

        if index < 1 or index > min(len(matches), 30):
            console.print("[red]Choose a number shown in the table.[/red]")
            continue

        show_prediction(matches.iloc[index - 1], split)


def search_by_hash(results, split):
    """Find commits using a full or partial commit hash."""
    query = Prompt.ask("Enter full or partial commit hash").strip()

    if not query:
        console.print("[yellow]Search cannot be empty.[/yellow]")
        return

    matches = results[
        results["commit_id"].astype(str).str.contains(
            query, case=False, regex=False, na=False
        )
    ].copy()

    console.print(f"Found {len(matches):,} matching commit(s).")
    inspect_search_results(matches, split)


def filter_by_project(results, split):
    """Select a project and browse its commits."""
    if "project" not in results.columns:
        console.print("[red]Project metadata is unavailable.[/red]")
        return

    projects = sorted(results["project"].dropna().astype(str).unique())

    if not projects:
        console.print("[yellow]No projects are available.[/yellow]")
        return

    console.print("\n[bold]Available projects[/bold]")
    for number, project in enumerate(projects, start=1):
        count = int((results["project"] == project).sum())
        console.print(f"{number:>3}. {project} ({count:,} commits)")

    choice = Prompt.ask(
        "Enter a project number, or 'b' to go back"
    ).strip()

    if choice.lower() in {"b", "back", "q"}:
        return

    try:
        index = int(choice)
    except ValueError:
        console.print("[red]Enter a valid project number.[/red]")
        return

    if index < 1 or index > len(projects):
        console.print("[red]Project number is out of range.[/red]")
        return

    project = projects[index - 1]
    matches = results[results["project"] == project].copy()

    console.print(
        f"\n[bold]{project}[/bold]: {len(matches):,} commits"
    )
    show_commit_list(matches, title=f"Commits — {project}")

    inspect = Prompt.ask(
        "Inspect a commit by its displayed number? (y/n)",
        choices=["y", "n"],
        default="n",
    )

    if inspect == "y":
        inspect_search_results(matches, split)


def browse_by_risk(results, split):
    """Browse commits ordered by blended estimated defect probability."""
    direction = Prompt.ask(
        "Sort order",
        choices=["highest", "lowest"],
        default="highest",
    )

    ascending = direction == "lowest"
    ordered = results.sort_values(
        "blend_probability", ascending=ascending
    ).reset_index(drop=True)

    show_commit_list(
        ordered,
        title=f"Commits — {direction} estimated risk",
    )
    inspect_search_results(ordered, split)


def main():
    parser = argparse.ArgumentParser(
        description="Offline Rich CLI for Experiment 7F."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Choose the canonical split (default: test).",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]Hybrid Feature Fusion — JIT Software "
            "Defect Prediction[/bold cyan]\n"
            "Experiment 7F | Offline demonstration",
            border_style="cyan",
        )
    )

    console.print(
        f"Loading the {args.split} split and running saved models..."
    )

    results = build_results(args.split)

    if results.empty:
        console.print("[red]No predictions were loaded.[/red]")
        return

    console.print(
        f"[green]Loaded predictions for {len(results):,} commits.[/green]"
    )

    while True:
        console.print(
            "\n[bold cyan]Main Menu[/bold cyan]\n"
            "1. Inspect commit by row number\n"
            "2. Search by commit hash\n"
            "3. Filter by project\n"
            "4. Browse highest/lowest estimated risk\n"
            "5. Inspect a random commit\n"
            "q. Quit"
        )

        choice = Prompt.ask("Choose an option").strip().lower()

        if choice in {"q", "quit", "exit"}:
            console.print("Demo finished.")
            break

        elif choice == "1":
            value = Prompt.ask(
                f"Enter a row number (1-{len(results):,})"
            ).strip()

            try:
                index = int(value)
            except ValueError:
                console.print("[red]Enter a valid number.[/red]")
                continue

            if index < 1 or index > len(results):
                console.print(
                    f"[red]Choose a number between 1 and {len(results)}.[/red]"
                )
                continue

            show_prediction(results.iloc[index - 1], args.split)

        elif choice == "2":
            search_by_hash(results, args.split)

        elif choice == "3":
            filter_by_project(results, args.split)

        elif choice == "4":
            browse_by_risk(results, args.split)

        elif choice == "5":
            show_prediction(results.sample(n=1).iloc[0], args.split)

        else:
            console.print("[red]Choose 1, 2, 3, 4, 5, or q.[/red]")


if __name__ == "__main__":
    main()
