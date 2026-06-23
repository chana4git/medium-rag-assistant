import csv
from pathlib import Path


INPUT_CSV = Path("data") / "medium-english-50mb.csv"
OUTPUT_CSV = Path("data") / "medium-english-50mb-with-ids.csv"


def add_article_ids(input_path: Path = INPUT_CSV, output_path: Path = OUTPUT_CSV) -> None:
    rows_loaded = 0
    first_three_rows = []

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row.")

        output_columns = ["article_id"] + reader.fieldnames

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()

            for row_number, row in enumerate(reader):
                row["article_id"] = str(row_number)
                writer.writerow(row)
                rows_loaded += 1

                if len(first_three_rows) < 3:
                    first_three_rows.append(row.copy())

    print(f"Rows loaded: {rows_loaded}")
    print(f"Column names: {', '.join(output_columns)}")
    print("\nFirst 3 rows:")

    for row in first_three_rows:
        print(
            f'article_id={row["article_id"]}, '
            f'title={row["title"]}, '
            f'authors={row["authors"]}'
        )

    print(f"\nSaved new CSV to: {output_path}")


if __name__ == "__main__":
    add_article_ids()
