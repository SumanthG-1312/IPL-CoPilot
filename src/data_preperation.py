from pathlib import Path
import os

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(
    os.getenv(
        "IPL_SOURCE_DATASET_PATH",
        PROJECT_ROOT / "data" / "ipl_ball_by_ball.csv",
    )
)
OUTPUT_PATH = Path(
    os.getenv(
        "IPL_DATASET_PATH",
        PROJECT_ROOT / "data" / "ipl_ball_by_ball_cleaned.csv",
    )
)

REQUIRED_COLUMNS = {
    "ID",
    "innings",
    "overs",
    "ballnumber",
    "batter",
    "bowler",
    "non-striker", 
    "extra_type",
    "batsman_run",
    "extras_run",
    "total_run",
    "non_boundary",
    "isWicketDelivery",
    "player_out",
    "kind",
    "fielders_involved",
    "BattingTeam",
}


def prepare_dataset(
    input_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    for column in (
        "batter",
        "bowler",
        "non-striker",
        "BattingTeam",
        "extra_type",
        "player_out",
        "kind",
        "fielders_involved",
    ):
        df[column] = df[column].fillna("None").astype(str).str.strip()

    df = df.drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    prepared = prepare_dataset()
    print(
        f"Prepared {len(prepared):,} rows at "
        f"{OUTPUT_PATH}"
    )
