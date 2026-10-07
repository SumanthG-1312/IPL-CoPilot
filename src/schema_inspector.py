import json
import sys
from pathlib import Path

import pandas as pd


def inspect_csv(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    columns = []

    for column in df.columns:

        series = df[column]

        sample_values = (
            series.dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "nullable": bool(series.isna().any()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": sample_values,
            }
        )

    return {
        "table_name": path.stem,
        "source_file": path.name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
    }


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python src/schema_inspector.py "
            "<csv_path>"
        )

        sys.exit(1)

    file_path = sys.argv[1]

    try:

        schema = inspect_csv(file_path)

        print(
            json.dumps(
                schema,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as error:

        print(
            f"Error: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()