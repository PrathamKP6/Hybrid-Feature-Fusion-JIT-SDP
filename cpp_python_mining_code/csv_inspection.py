import csv
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CHANGE ONLY THIS
# ============================================================
INPUT_FILE = r"D:\embeddings_and_pca_code\data\java_only_latest_36k_pca384.csv"


# ============================================================
# Parse author_date
# ============================================================
def parse_author_date(value):
    if not value:
        return None

    value = value.strip()

    # ---- Unix timestamp ----
    try:
        timestamp = int(float(value))

        # Ignore obviously invalid timestamps
        if 0 < timestamp < 4102444800:  # before year 2100
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass

    # ---- ISO-8601 date ----
    try:
        # Handle Z timezone
        value_iso = value.replace("Z", "+00:00")

        dt = datetime.fromisoformat(value_iso)

        # If timezone is missing, treat as UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


# ============================================================
# Main
# ============================================================
def find_date_range(filename):
    earliest = None
    latest = None

    total_rows = 0
    valid_dates = 0
    invalid_dates = 0

    # Important for huge diff columns
    csv.field_size_limit(2**31 - 1)

    path = Path(filename)

    if not path.exists():
        print(f"\nERROR: File not found:\n{path}")
        return

    print(f"\nReading:\n{path}")
    print("Please wait...\n")

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            print("ERROR: Could not detect CSV header.")
            return

        # Find author_date column safely
        author_column = None

        for column in reader.fieldnames:
            if column and column.strip().lower() == "author_date":
                author_column = column
                break

        if author_column is None:
            print("ERROR: 'author_date' column not found.")
            print("\nDetected columns:")
            for column in reader.fieldnames:
                print(f"  {column}")
            return

        for row in reader:
            total_rows += 1

            raw_date = row.get(author_column, "")

            dt = parse_author_date(raw_date)

            if dt is None:
                invalid_dates += 1
                continue

            valid_dates += 1

            if earliest is None or dt < earliest:
                earliest = dt

            if latest is None or dt > latest:
                latest = dt

    # ========================================================
    # Results
    # ========================================================
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Total commit records : {total_rows}")
    print(f"Valid dates          : {valid_dates}")
    print(f"Invalid/missing dates: {invalid_dates}")

    if earliest is None:
        print("\nNo valid author_date values found.")
        return

    print("\nEarliest author date:")
    print(earliest.strftime("%B %d, %Y"))

    print("\nLatest author date:")
    print(latest.strftime("%B %d, %Y"))

    print("\nOverall range:")
    print(
        f"{earliest.strftime('%B %Y')} "
        f"→ "
        f"{latest.strftime('%B %Y')}"
    )

    print("=" * 60)


if __name__ == "__main__":
    find_date_range(INPUT_FILE)