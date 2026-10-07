import pandas as pd

from metric_engine import calculate_metric


DATA_PATH = "data/ipl_ball_by_ball_cleaned.csv"


df = pd.read_csv(
    DATA_PATH,
    keep_default_na=False,
    na_filter=False
)


def player_data(
    player: str,
    column: str
):
    return df[
        df[column].astype(str).str.lower()
        == player.lower()
    ]


tests = [
    (
        "G Gambhir runs",
        player_data("G Gambhir", "batter"),
        "runs",
        "batter"
    ),
    (
        "G Gambhir dismissals",
        player_data("G Gambhir", "batter"),
        "dismissals",
        "batter"
    ),
    (
        "Mohammed Siraj wickets",
        player_data("Mohammed Siraj", "bowler"),
        "wickets",
        "bowler"
    ),
    (
        "Mohammed Siraj balls bowled",
        player_data("Mohammed Siraj", "bowler"),
        "balls_bowled",
        "bowler"
    ),
    (
        "G Gambhir fours",
        player_data("G Gambhir", "batter"),
        "fours",
        "batter"
    ),
    (
        "G Gambhir sixes",
        player_data("G Gambhir", "batter"),
        "sixes",
        "batter"
    ),
    (
        "G Gambhir strike rate",
        player_data("G Gambhir", "batter"),
        "strike_rate",
        "batter"
    ),
    (
        "Mohammed Siraj economy",
        player_data("Mohammed Siraj", "bowler"),
        "economy",
        "bowler"
    ),
]


for name, data, metric, role in tests:

    try:

        result = calculate_metric(
            data,
            metric,
            role
        )

        print(f"{name}: {result}")

    except Exception as error:

        print(f"{name}: ERROR -> {error}")
        