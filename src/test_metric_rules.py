import pandas as pd

from metric_engine import (
    calculate_metric,
)


DATA_PATH = "data/ipl_ball_by_ball_cleaned.csv"


df = pd.read_csv(
    DATA_PATH,
    keep_default_na=False,
    na_filter=False
)


def player_data(player, column):
    return df[
        df[column].astype(str).str.lower()
        == player.lower()
    ]


def check(condition, message):
    if condition:
        print(f"PASS: {message}")
    else:
        print(f"FAIL: {message}")


# Basic dataset checks

total_rows = len(df)

check(
    total_rows == 225954,
    "Dataset contains 225,954 rows"
)

check(
    df["overs"].between(0, 19).all(),
    "Overs are zero-based from 0 to 19"
)

check(
    df["overs"].between(0, 19).all(),
    "Overs are zero-based from 0 to 19"
)


# Gambhir batting rules

gambhir = player_data(
    "G Gambhir",
    "batter"
)

runs = calculate_metric(
    gambhir,
    "runs",
    "batter"
)

fours = calculate_metric(
    gambhir,
    "fours",
    "batter"
)

sixes = calculate_metric(
    gambhir,
    "sixes",
    "batter"
)

balls_faced = calculate_metric(
    gambhir,
    "balls_faced",
    "batter"
)

dismissals = calculate_metric(
    gambhir,
    "dismissals",
    "batter"
)

strike_rate = calculate_metric(
    gambhir,
    "strike_rate",
    "batter"
)


check(
    runs >= 0,
    "Batting runs are non-negative"
)

check(
    fours >= 0 and sixes >= 0,
    "Boundary counts are non-negative"
)

check(
    fours + sixes <= balls_faced,
    "Boundaries cannot exceed balls faced"
)

check(
    dismissals >= 0,
    "Dismissals are non-negative"
)

if balls_faced > 0:

    expected_strike_rate = (
        runs / balls_faced * 100
    )

    check(
        abs(
            strike_rate
            - expected_strike_rate
        ) < 1e-9,
        "Strike rate formula is correct"
    )


# Siraj bowling rules

siraj = player_data(
    "Mohammed Siraj",
    "bowler"
)

wickets = calculate_metric(
    siraj,
    "wickets",
    "bowler"
)

balls_bowled = calculate_metric(
    siraj,
    "balls_bowled",
    "bowler"
)

runs_conceded = calculate_metric(
    siraj,
    "runs_conceded",
    "bowler"
)

economy = calculate_metric(
    siraj,
    "economy",
    "bowler"
)


check(
    wickets >= 0,
    "Bowler wickets are non-negative"
)

check(
    balls_bowled >= 0,
    "Balls bowled are non-negative"
)

check(
    runs_conceded >= 0,
    "Runs conceded are non-negative"
)

if balls_bowled > 0:

    expected_economy = (
        runs_conceded
        / balls_bowled
        * 6
    )

    check(
        abs(
            economy
            - expected_economy
        ) < 1e-9,
        "Economy formula is correct"
    )


# Wicket-credit validation

wicket_rows = df[
    df["isWicketDelivery"].astype(str)
    .isin(["1", "1.0", "True", "true"])
]

kind = (
    wicket_rows["kind"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

non_bowler_kinds = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}

bowler_credit_wickets = (
    ~kind.isin(non_bowler_kinds)
).sum()


calculated_total_wickets = (
    calculate_metric(
        df,
        "wickets",
        "bowler"
    )
)


check(
    calculated_total_wickets
    == bowler_credit_wickets,
    "Bowler wicket-credit rule matches metric engine"
)


# Boundary validation

boundary_rows = gambhir[
    gambhir["batsman_run"].isin([4, 6])
]

non_boundary_rows = boundary_rows[
    boundary_rows["non_boundary"].astype(str).str.lower()
    .isin(["1", "1.0", "true", "yes"])
]

calculated_boundaries = (
    fours + sixes
)

expected_boundaries = (
    len(boundary_rows)
    - len(non_boundary_rows)
)


check(
    calculated_boundaries
    == expected_boundaries,
    "Boundary calculation respects non_boundary"
)


print("\nMetric rule validation complete.")