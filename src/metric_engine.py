from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


MetricFunction = Callable[[pd.DataFrame, str], float]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    roles: frozenset[str]
    function: MetricFunction


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


NON_BOWLER_WICKET_KINDS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}


WIDE_VALUES = {
    "wide",
    "wides",
}


NO_BALL_VALUES = {
    "no ball",
    "no-ball",
    "noball",
    "noballs",
    "no balls",
}


BYE_VALUES = {
    "bye",
    "byes",
}


LEG_BYE_VALUES = {
    "leg bye",
    "leg-bye",
    "legbye",
    "legbyes",
    "leg byes",
}


PENALTY_VALUES = {
    "penalty",
    "penalty runs",
}


def validate_dataframe(df: pd.DataFrame) -> None:

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _normalized_text(series: pd.Series) -> pd.Series:

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )


def _truthy(series: pd.Series) -> pd.Series:

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return (
            series
            .fillna(0)
            .astype(float)
            .ne(0)
        )

    normalized = _normalized_text(series)

    return normalized.isin({
        "1",
        "true",
        "yes",
        "y",
    })


def _legal_delivery_mask(
    df: pd.DataFrame
) -> pd.Series:

    extra_type = _normalized_text(
        df["extra_type"]
    )

    illegal_values = (
        WIDE_VALUES
        | NO_BALL_VALUES
    )

    return ~extra_type.isin(
        illegal_values
    )


def _wicket_event_mask(
    df: pd.DataFrame
) -> pd.Series:

    return _truthy(
        df["isWicketDelivery"]
    )


def _bowler_wicket_mask(
    df: pd.DataFrame
) -> pd.Series:

    kind = _normalized_text(
        df["kind"]
    )

    non_bowler_wicket = kind.isin(
        NON_BOWLER_WICKET_KINDS
    )

    return (
        _wicket_event_mask(df)
        & ~non_bowler_wicket
    )


def _player_out_mask(
    df: pd.DataFrame
) -> pd.Series:

    player_out = _normalized_text(
        df["player_out"]
    )

    return ~player_out.isin({
        "",
        "none",
        "nan",
        "null",
    })


def _boundary_mask(
    df: pd.DataFrame,
    run_value: int
) -> pd.Series:

    batsman_run = pd.to_numeric(
        df["batsman_run"],
        errors="coerce"
    ).fillna(0)

    return (
        batsman_run.eq(run_value)
        & ~_truthy(df["non_boundary"])
    )


def _metric_runs(
    df: pd.DataFrame,
    role: str
) -> float:

    if role == "bowler":
        return _metric_runs_conceded(
            df,
            role
        )

    if role == "team":

        return float(
            pd.to_numeric(
                df["total_run"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    return float(
        pd.to_numeric(
            df["batsman_run"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


def _metric_fours(
    df: pd.DataFrame,
    role: str
) -> float:

    if role not in {
        "batter",
        "team",
    }:
        raise ValueError(
            "fours requires batter or team role"
        )

    return float(
        _boundary_mask(
            df,
            4
        ).sum()
    )


def _metric_sixes(
    df: pd.DataFrame,
    role: str
) -> float:

    if role not in {
        "batter",
        "team",
    }:
        raise ValueError(
            "sixes requires batter or team role"
        )

    return float(
        _boundary_mask(
            df,
            6
        ).sum()
    )


def _metric_boundaries(
    df: pd.DataFrame,
    role: str
) -> float:

    return (
        _metric_fours(df, role)
        + _metric_sixes(df, role)
    )


def _metric_balls_faced(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "batter":
        raise ValueError(
            "balls_faced requires batter role"
        )

    return float(
        _legal_delivery_mask(df).sum()
    )


def _metric_balls_bowled(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "bowler":
        raise ValueError(
            "balls_bowled requires bowler role"
        )

    return float(
        _legal_delivery_mask(df).sum()
    )


def _metric_dismissals(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "batter":
        raise ValueError(
            "dismissals requires batter role"
        )

    return float(
        _player_out_mask(df).sum()
    )


def _metric_wickets(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "bowler":
        raise ValueError(
            "wickets requires bowler role"
        )

    return float(
        _bowler_wicket_mask(df).sum()
    )


def _metric_runs_conceded(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "bowler":
        raise ValueError(
            "runs_conceded requires bowler role"
        )

    total_run = pd.to_numeric(
        df["total_run"],
        errors="coerce"
    ).fillna(0)

    batsman_run = pd.to_numeric(
        df["batsman_run"],
        errors="coerce"
    ).fillna(0)

    extra_type = _normalized_text(
        df["extra_type"]
    )

    excluded = extra_type.isin(
        BYE_VALUES
        | LEG_BYE_VALUES
        | PENALTY_VALUES
    )

    conceded = total_run.where(
        ~excluded,
        batsman_run
    )

    return float(
        conceded.sum()
    )


def _metric_strike_rate(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "batter":
        raise ValueError(
            "strike_rate requires batter role"
        )

    runs = _metric_runs(
        df,
        "batter"
    )

    balls = _metric_balls_faced(
        df,
        "batter"
    )

    if balls == 0:
        return float("nan")

    return float(
        runs / balls * 100
    )


def _metric_economy(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "bowler":
        raise ValueError(
            "economy requires bowler role"
        )

    runs = _metric_runs_conceded(
        df,
        "bowler"
    )

    balls = _metric_balls_bowled(
        df,
        "bowler"
    )

    if balls == 0:
        return float("nan")

    return float(
        runs / balls * 6
    )


def _metric_batting_average(
    df: pd.DataFrame,
    role: str
) -> float:

    runs = _metric_runs(
        df,
        "batter"
    )

    dismissals = _metric_dismissals(
        df,
        "batter"
    )

    if dismissals == 0:
        return float("nan")

    return float(
        runs / dismissals
    )


def _metric_bowling_average(
    df: pd.DataFrame,
    role: str
) -> float:

    runs = _metric_runs_conceded(
        df,
        "bowler"
    )

    wickets = _metric_wickets(
        df,
        "bowler"
    )

    if wickets == 0:
        return float("nan")

    return float(
        runs / wickets
    )


def _metric_dot_balls(
    df: pd.DataFrame,
    role: str
) -> float:

    if role not in {
        "batter",
        "bowler",
        "team",
    }:
        raise ValueError(
            "dot_balls requires batter, bowler or team role"
        )

    total_run = pd.to_numeric(
        df["total_run"],
        errors="coerce"
    ).fillna(0)

    return float(
        (
            _legal_delivery_mask(df)
            & total_run.eq(0)
        ).sum()
    )


def _metric_team_extras(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "team":
        raise ValueError(
            "team_extras requires team role"
        )

    return float(
        pd.to_numeric(
            df["extras_run"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


def _metric_team_wickets(
    df: pd.DataFrame,
    role: str
) -> float:

    if role != "team":
        raise ValueError(
            "team_wickets requires team role"
        )

    return float(
        _wicket_event_mask(df).sum()
    )


def _metric_run_rate(
    df: pd.DataFrame,
    role: str
) -> float:

    total_runs = float(
        pd.to_numeric(
            df["total_run"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

    legal_balls = float(
        _legal_delivery_mask(df).sum()
    )

    if legal_balls == 0:
        return float("nan")

    return float(
        total_runs / legal_balls * 6
    )


def _metric_match_runs(
    df: pd.DataFrame,
    role: str
) -> float:

    return float(
        pd.to_numeric(
            df["total_run"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


def _metric_match_wickets(
    df: pd.DataFrame,
    role: str
) -> float:

    return float(
        _wicket_event_mask(df).sum()
    )


METRIC_REGISTRY = {

    "runs": MetricDefinition(
        "runs",
        frozenset({
            "batter",
            "bowler",
            "team",
        }),
        _metric_runs
    ),

    "fours": MetricDefinition(
        "fours",
        frozenset({
            "batter",
            "team",
        }),
        _metric_fours
    ),

    "sixes": MetricDefinition(
        "sixes",
        frozenset({
            "batter",
            "team",
        }),
        _metric_sixes
    ),

    "boundaries": MetricDefinition(
        "boundaries",
        frozenset({
            "batter",
            "team",
        }),
        _metric_boundaries
    ),

    "balls_faced": MetricDefinition(
        "balls_faced",
        frozenset({"batter"}),
        _metric_balls_faced
    ),

    "balls_bowled": MetricDefinition(
        "balls_bowled",
        frozenset({"bowler"}),
        _metric_balls_bowled
    ),

    "strike_rate": MetricDefinition(
        "strike_rate",
        frozenset({"batter"}),
        _metric_strike_rate
    ),

    "dismissals": MetricDefinition(
        "dismissals",
        frozenset({"batter"}),
        _metric_dismissals
    ),

    "wickets": MetricDefinition(
        "wickets",
        frozenset({"bowler"}),
        _metric_wickets
    ),

    "runs_conceded": MetricDefinition(
        "runs_conceded",
        frozenset({"bowler"}),
        _metric_runs_conceded
    ),

    "economy": MetricDefinition(
        "economy",
        frozenset({"bowler"}),
        _metric_economy
    ),

    "batting_average": MetricDefinition(
        "batting_average",
        frozenset({"batter"}),
        _metric_batting_average
    ),

    "bowling_average": MetricDefinition(
        "bowling_average",
        frozenset({"bowler"}),
        _metric_bowling_average
    ),

    "dot_balls": MetricDefinition(
        "dot_balls",
        frozenset({
            "batter",
            "bowler",
            "team",
        }),
        _metric_dot_balls
    ),

    "team_runs": MetricDefinition(
        "team_runs",
        frozenset({"team"}),
        _metric_runs
    ),

    "team_extras": MetricDefinition(
        "team_extras",
        frozenset({"team"}),
        _metric_team_extras
    ),

    "team_fours": MetricDefinition(
        "team_fours",
        frozenset({"team"}),
        _metric_fours
    ),

    "team_sixes": MetricDefinition(
        "team_sixes",
        frozenset({"team"}),
        _metric_sixes
    ),

    "team_wickets": MetricDefinition(
        "team_wickets",
        frozenset({"team"}),
        _metric_team_wickets
    ),

    "run_rate": MetricDefinition(
        "run_rate",
        frozenset({
            "team",
            "match",
            "innings",
        }),
        _metric_run_rate
    ),

    "match_runs": MetricDefinition(
        "match_runs",
        frozenset({
            "match",
            "innings",
        }),
        _metric_match_runs
    ),

    "match_wickets": MetricDefinition(
        "match_wickets",
        frozenset({
            "match",
            "innings",
        }),
        _metric_match_wickets
    ),
}


def get_metric_definition(
    metric_name: str
) -> MetricDefinition:

    if metric_name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unsupported metric: {metric_name}"
        )

    return METRIC_REGISTRY[metric_name]


def calculate_metric(
    df: pd.DataFrame,
    metric_name: str,
    role: str
) -> float:

    validate_dataframe(df)

    definition = get_metric_definition(
        metric_name
    )

    if role not in definition.roles:
        raise ValueError(
            f"Metric '{metric_name}' is not supported "
            f"for role '{role}'"
        )

    return definition.function(
        df,
        role
    )


def calculate_grouped_metric(
    df: pd.DataFrame,
    metric_name: str,
    group_column: str,
    role: str
) -> pd.DataFrame:

    validate_dataframe(df)

    if group_column not in df.columns:
        raise ValueError(
            f"Unknown group column: {group_column}"
        )

    rows = []

    for group_value, group in df.groupby(
        group_column,
        dropna=False
    ):

        value = calculate_metric(
            group,
            metric_name,
            role
        )

        rows.append({
            group_column: group_value,
            metric_name: value
        })

    return pd.DataFrame(rows)


def list_metrics() -> list[str]:
    return sorted(
        METRIC_REGISTRY.keys()
    )