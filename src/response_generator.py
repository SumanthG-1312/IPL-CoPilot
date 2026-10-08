from __future__ import annotations

import os
import re
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )

MODEL = os.getenv("GROQ_MODEL")

if not MODEL:
    raise ValueError(
        "GROQ_MODEL is missing from .env"
    )


# ============================================================
# LANGCHAIN GROQ
# ============================================================

llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=700,
    reasoning_effort="none",
)


# ============================================================
# METRIC CONFIGURATION
# ============================================================

METRIC_LABELS = {
    "runs": "Runs",
    "fours": "Fours",
    "sixes": "Sixes",
    "boundaries": "Boundaries",
    "balls_faced": "Balls Faced",
    "strike_rate": "Strike Rate",
    "dismissals": "Dismissals",
    "batting_average": "Batting Average",
    "wickets": "Wickets",
    "balls_bowled": "Balls Bowled",
    "runs_conceded": "Runs Conceded",
    "economy": "Economy",
    "bowling_average": "Bowling Average",
    "dot_balls": "Dot Balls",
    "partnership_runs": "Partnership Runs",
}


METRIC_UNITS = {
    "runs": "runs",
    "fours": "fours",
    "sixes": "sixes",
    "boundaries": "boundaries",
    "balls_faced": "balls",
    "strike_rate": "",
    "dismissals": "dismissals",
    "batting_average": "",
    "wickets": "wickets",
    "balls_bowled": "balls",
    "runs_conceded": "runs",
    "economy": "",
    "bowling_average": "",
    "dot_balls": "dot balls",
    "partnership_runs": "runs",
}


# ============================================================
# RESULT COLUMN ALIASES
# ============================================================

METRIC_COLUMN_ALIASES = {
    "runs": [
        "runs",
        "total_runs",
        "total_run",
        "run",
    ],
    "fours": [
        "fours",
        "total_fours",
        "total_four",
        "four",
    ],
    "sixes": [
        "sixes",
        "total_sixes",
        "total_six",
        "six",
    ],
    "boundaries": [
        "boundaries",
        "total_boundaries",
        "boundary",
    ],
    "balls_faced": [
        "balls_faced",
        "total_balls_faced",
        "balls_faced_count",
    ],
    "strike_rate": [
        "strike_rate",
        "strike rate",
    ],
    "dismissals": [
        "dismissals",
        "total_dismissals",
        "dismissal",
    ],
    "batting_average": [
        "batting_average",
        "batting average",
    ],
    "wickets": [
        "wickets",
        "total_wickets",
        "wicket",
    ],
    "balls_bowled": [
        "balls_bowled",
        "total_balls_bowled",
        "balls_bowled_count",
    ],
    "runs_conceded": [
        "runs_conceded",
        "total_runs_conceded",
        "runs_conceded_count",
    ],
    "economy": [
        "economy",
        "economy_rate",
        "economy rate",
    ],
    "bowling_average": [
        "bowling_average",
        "bowling average",
    ],
    "dot_balls": [
        "dot_balls",
        "total_dot_balls",
        "dot_ball",
    ],
    "partnership_runs": [
        "partnership_runs",
        "total_partnership_runs",
    ],
}


# ============================================================
# NAME COLUMN ALIASES
# ============================================================

PLAYER_COLUMNS = [
    "player",
    "batter",
    "bowler",
    "name",
    "player_name",
    "player_name_1",
    "Player",
    "Batter",
    "Bowler",
]


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for comparisons and column lookup.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = value.replace(
        "-",
        " ",
    )

    value = value.replace(
        "_",
        " ",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def is_missing_value(value: Any) -> bool:
    """
    Safely determine whether a value is missing.
    """

    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

    except Exception:
        pass

    return False


# ============================================================
# NUMBER FORMATTING
# ============================================================

def format_number(
    value: Any,
    metric: Optional[str] = None,
) -> str:
    """
    Format numbers according to the metric.

    Examples:
        6634 -> 6,634
        129.7 -> 129.70
    """

    if is_missing_value(value):
        return "—"

    try:
        numeric_value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    decimal_metrics = {
        "strike_rate",
        "batting_average",
        "economy",
        "bowling_average",
    }

    if metric in decimal_metrics:
        return f"{numeric_value:,.2f}"

    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"

    return f"{numeric_value:,.2f}"


# ============================================================
# RESULT COLUMN LOOKUP
# ============================================================

def get_column_map(
    row: pd.Series,
) -> dict[str, str]:
    """
    Build case-insensitive normalized column mapping.
    """

    mapping: dict[str, str] = {}

    for column in row.index:

        normalized = normalize_text(
            column
        )

        mapping[normalized] = column

    return mapping


def get_metric_value(
    row: pd.Series,
    metric: str,
) -> Any:
    """
    Extract a metric from a result row.

    Supports both:
        runs
        total_runs

    and other planner-generated aliases.
    """

    column_map = get_column_map(
        row
    )

    candidates = METRIC_COLUMN_ALIASES.get(
        metric,
        [metric],
    )

    for candidate in candidates:

        normalized_candidate = normalize_text(
            candidate
        )

        actual_column = column_map.get(
            normalized_candidate
        )

        if actual_column is None:
            continue

        value = row[actual_column]

        if is_missing_value(value):
            continue

        return value

    return None


# ============================================================
# ENTITY COLUMN LOOKUP
# ============================================================

def get_entity_value(
    row: pd.Series,
) -> Optional[str]:
    """
    Find the player/entity column from a result row.
    """

    column_map = get_column_map(
        row
    )

    for candidate in PLAYER_COLUMNS:

        actual_column = column_map.get(
            normalize_text(candidate)
        )

        if actual_column is None:
            continue

        value = row[actual_column]

        if is_missing_value(value):
            continue

        return str(value)

    # Fallback:
    # look for columns containing "player"
    for normalized_name, actual_column in column_map.items():

        if "player" not in normalized_name:
            continue

        value = row[actual_column]

        if is_missing_value(value):
            continue

        return str(value)

    return None


# ============================================================
# QUESTION HELPERS
# ============================================================

def get_response_type(
    question: str,
) -> str:
    """
    Determine whether the question requires:
        simple
        comparison
        ranking
    """

    normalized = normalize_text(
        question
    )

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bvs\.\b",
        r"\bbetween\b",
    ]

    for pattern in comparison_patterns:

        if re.search(
            pattern,
            normalized,
        ):
            return "comparison"

    ranking_patterns = [
        r"\btop\s+\d+",
        r"\btop\b",
        r"\branking\b",
        r"\brank\b",
        r"\bmost\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bleast\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\bwho scored the most\b",
        r"\bwho took the most\b",
    ]

    for pattern in ranking_patterns:

        if re.search(
            pattern,
            normalized,
        ):
            return "ranking"

    return "simple"


# ============================================================
# METRIC DETECTION FROM QUESTION
# ============================================================

def detect_metrics_from_question(
    question: str,
) -> list[str]:
    """
    Detect metrics mentioned in the question.

    Used as a fallback when the planner does not explicitly
    provide metrics.
    """

    normalized = normalize_text(
        question
    )

    metric_phrases = {
        "strike_rate": [
            "strike rate",
            "strike-rate",
            "strike_rate",
        ],
        "batting_average": [
            "batting average",
            "batting-average",
        ],
        "bowling_average": [
            "bowling average",
            "bowling-average",
        ],
        "runs_conceded": [
            "runs conceded",
            "runs given",
            "runs gave",
        ],
        "balls_faced": [
            "balls faced",
            "balls face",
        ],
        "balls_bowled": [
            "balls bowled",
            "balls bowl",
        ],
        "dot_balls": [
            "dot balls",
            "dot ball",
        ],
        "dismissals": [
            "dismissals",
            "dismissed",
            "got out",
            "get out",
            "was out",
        ],
        "boundaries": [
            "boundaries",
            "boundary",
        ],
        "sixes": [
            "sixes",
            "sixers",
            "sixer",
            "six hitters",
            "six hitter",
        ],
        "fours": [
            "fours",
            "four",
        ],
        "wickets": [
            "wickets",
            "wicket",
        ],
        "economy": [
            "economy rate",
            "economy",
        ],
        "partnership_runs": [
            "partnership runs",
            "partnership",
        ],
        "runs": [
            "runs",
            "run",
        ],
    }

    matches = []

    for metric, phrases in metric_phrases.items():

        for phrase in phrases:

            normalized_phrase = normalize_text(
                phrase
            )

            match = re.search(
                rf"\b{re.escape(normalized_phrase)}\b",
                normalized,
            )

            if match:
                matches.append(
                    (
                        match.start(),
                        match.end(),
                        metric,
                    )
                )

    # Longest phrase first.
    matches.sort(
        key=lambda item: (
            item[1] - item[0],
            -item[0],
        ),
        reverse=True,
    )

    detected: list[str] = []
    occupied: list[tuple[int, int]] = []

    for start, end, metric in matches:

        overlaps = any(
            start < other_end
            and end > other_start
            for other_start, other_end
            in occupied
        )

        if overlaps:
            continue

        occupied.append(
            (
                start,
                end,
            )
        )

        if metric not in detected:
            detected.append(metric)

    # "partnership runs" should not also become "runs".
    if (
        "partnership_runs" in detected
        and "runs" in detected
    ):
        detected.remove("runs")

    return detected


# ============================================================
# SIMPLE RESPONSE
# ============================================================

def format_metric_line(
    metric: str,
    value: Any,
) -> str:
    """
    Format one metric as a readable line.
    """

    label = METRIC_LABELS.get(
        metric,
        metric.replace(
            "_",
            " ",
        ).title(),
    )

    formatted_value = format_number(
        value,
        metric,
    )

    unit = METRIC_UNITS.get(
        metric,
        "",
    )

    if unit:
        return (
            f"**{label}:** "
            f"{formatted_value} {unit}"
        )

    return (
        f"**{label}:** "
        f"{formatted_value}"
    )


def format_simple_response(
    question: str,
    result: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> str:
    """
    Format a normal single-player/statistic response.
    """

    if result is None or result.empty:
        return "No matching data was found."

    metrics = metrics or detect_metrics_from_question(
        question
    )

    # --------------------------------------------------------
    # One row
    # --------------------------------------------------------

    if len(result) == 1:

        row = result.iloc[0]

        entity = get_entity_value(
            row
        )

        # ----------------------------------------------------
        # No explicit entity column
        # ----------------------------------------------------

        if not entity:

            # If there is a single metric, show it directly.
            if len(metrics) == 1:

                metric = metrics[0]

                value = get_metric_value(
                    row,
                    metric,
                )

                if value is None:
                    return "No matching data was found."

                label = METRIC_LABELS.get(
                    metric,
                    metric.title(),
                )

                formatted = format_number(
                    value,
                    metric,
                )

                unit = METRIC_UNITS.get(
                    metric,
                    "",
                )

                if unit:
                    return (
                        f"### {label}\n\n"
                        f"**{formatted} {unit}**"
                    )

                return (
                    f"### {label}\n\n"
                    f"**{formatted}**"
                )

            lines = []

            for metric in metrics:

                value = get_metric_value(
                    row,
                    metric,
                )

                if value is None:
                    continue

                lines.append(
                    format_metric_line(
                        metric,
                        value,
                    )
                )

            if not lines:
                return "No matching data was found."

            return "\n".join(lines)

        # ----------------------------------------------------
        # Single entity + single metric
        # ----------------------------------------------------

        if len(metrics) == 1:

            metric = metrics[0]

            value = get_metric_value(
                row,
                metric,
            )

            if value is None:
                return (
                    f"### {entity}\n\n"
                    "No matching data was found."
                )

            formatted = format_number(
                value,
                metric,
            )

            unit = METRIC_UNITS.get(
                metric,
                "",
            )

            if unit:
                return (
                    f"### {entity}\n\n"
                    f"**{formatted} {unit}**"
                )

            return (
                f"### {entity}\n\n"
                f"**{formatted}**"
            )

        # ----------------------------------------------------
        # Single entity + multiple metrics
        # ----------------------------------------------------

        lines = []

        for metric in metrics:

            value = get_metric_value(
                row,
                metric,
            )

            if value is None:
                continue

            lines.append(
                format_metric_line(
                    metric,
                    value,
                )
            )

        if not lines:
            return (
                f"### {entity}\n\n"
                "No matching data was found."
            )

        return (
            f"### {entity}\n\n"
            + "\n".join(lines)
        )

    # --------------------------------------------------------
    # Multiple rows without ranking/comparison
    # --------------------------------------------------------

    lines = []

    for _, row in result.iterrows():

        entity = get_entity_value(
            row
        )

        if not entity:
            entity = "Result"

        if len(metrics) == 1:

            metric = metrics[0]

            value = get_metric_value(
                row,
                metric,
            )

            if value is None:
                continue

            formatted = format_number(
                value,
                metric,
            )

            unit = METRIC_UNITS.get(
                metric,
                "",
            )

            if unit:
                lines.append(
                    f"**{entity}:** "
                    f"{formatted} {unit}"
                )
            else:
                lines.append(
                    f"**{entity}:** "
                    f"{formatted}"
                )

        else:

            metric_parts = []

            for metric in metrics:

                value = get_metric_value(
                    row,
                    metric,
                )

                if value is None:
                    continue

                label = METRIC_LABELS.get(
                    metric,
                    metric.title(),
                )

                formatted = format_number(
                    value,
                    metric,
                )

                metric_parts.append(
                    f"{label}: {formatted}"
                )

            if metric_parts:

                lines.append(
                    f"**{entity}:** "
                    + ", ".join(metric_parts)
                )

    if not lines:
        return "No matching data was found."

    return "\n".join(lines)


# ============================================================
# COMPARISON RESPONSE
# ============================================================

def format_comparison_response(
    result: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> str:
    """
    Format player comparison as a Markdown table.
    """

    if result is None or result.empty:
        return "No comparison data was found."

    metrics = metrics or []

    if not metrics:
        metrics = [
            metric
            for metric in METRIC_LABELS
            if any(
                normalize_text(column)
                in {
                    normalize_text(alias)
                    for alias
                    in METRIC_COLUMN_ALIASES.get(
                        metric,
                        [],
                    )
                }
                for column in result.columns
            )
        ]

    if not metrics:
        return "No comparison metrics were found."

    header = [
        "Player"
    ]

    for metric in metrics:

        header.append(
            METRIC_LABELS.get(
                metric,
                metric.title(),
            )
        )

    lines = []

    lines.append(
        "| "
        + " | ".join(header)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            ["---"] + ["---:"] * len(metrics)
        )
        + " |"
    )

    for _, row in result.iterrows():

        player = get_entity_value(
            row
        )

        if not player:
            player = "Unknown"

        values = [
            player
        ]

        for metric in metrics:

            value = get_metric_value(
                row,
                metric,
            )

            values.append(
                format_number(
                    value,
                    metric,
                )
            )

        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )

    return (
        "### Player Comparison\n\n"
        + "\n".join(lines)
    )


# ============================================================
# RANKING HELPERS
# ============================================================

def extract_rank_from_result(
    row: pd.Series,
    fallback_rank: int,
) -> int:
    """
    Use an existing Rank column when available.
    """

    column_map = get_column_map(
        row
    )

    rank_columns = [
        "rank",
        "ranking",
        "position",
    ]

    for candidate in rank_columns:

        actual_column = column_map.get(
            normalize_text(candidate)
        )

        if actual_column is None:
            continue

        value = row[actual_column]

        if is_missing_value(value):
            continue

        try:
            return int(float(value))
        except (
            TypeError,
            ValueError,
        ):
            continue

    return fallback_rank


def get_ranking_metric(
    question: str,
    metrics: list[str],
) -> Optional[str]:
    """
    Determine the primary ranking metric.
    """

    if metrics:
        return metrics[0]

    normalized = normalize_text(
        question
    )

    if "wicket" in normalized:
        return "wickets"

    if "six" in normalized:
        return "sixes"

    if "four" in normalized:
        return "fours"

    if "strike rate" in normalized:
        return "strike_rate"

    if "economy" in normalized:
        return "economy"

    if "runs" in normalized or "run" in normalized:
        return "runs"

    return None


def extract_top_n(
    question: str,
) -> Optional[int]:
    """
    Extract an explicit Top N.

    Examples:
        Top 5 batters -> 5
        Top 10 run scorers -> 10
    """

    match = re.search(
        r"\btop\s+(\d+)\b",
        normalize_text(question),
    )

    if match:
        try:
            return int(
                match.group(1)
            )
        except ValueError:
            return None

    return None


def build_ranking_title(
    question: str,
    metric: Optional[str],
) -> str:
    """
    Build a readable ranking title.
    """

    top_n = extract_top_n(
        question
    )

    if metric == "runs":
        subject = "Run Scorers"

    elif metric == "wickets":
        subject = "Wicket Takers"

    elif metric == "sixes":
        subject = "Six Hitters"

    elif metric == "fours":
        subject = "Four Hitters"

    elif metric == "strike_rate":
        subject = "Strike Rate"

    elif metric == "economy":
        subject = "Economy Rate"

    elif metric:
        subject = METRIC_LABELS.get(
            metric,
            metric.title(),
        )

    else:
        subject = "Ranking"

    if top_n:
        return f"### Top {top_n} {subject}"

    return f"### {subject}"


# ============================================================
# RANKING RESPONSE
# ============================================================

def format_ranking_response(
    question: str,
    result: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> str:
    """
    Format ranking queries as a Markdown table.
    """

    if result is None or result.empty:
        return "No ranking data was found."

    metrics = metrics or detect_metrics_from_question(
        question
    )

    metric = get_ranking_metric(
        question,
        metrics,
    )

    if metric is None:
        metric = "runs"

    title = build_ranking_title(
        question,
        metric,
    )

    lines = []

    lines.append(
        "| Rank | Player | "
        + METRIC_LABELS.get(
            metric,
            metric.title(),
        )
        + " |"
    )

    lines.append(
        "|---:|---|---:|"
    )

    for index, (_, row) in enumerate(
        result.iterrows(),
        start=1,
    ):

        rank = extract_rank_from_result(
            row,
            index,
        )

        player = get_entity_value(
            row
        )

        if not player:

            player_column_map = get_column_map(
                row
            )

            # Search for likely group column.
            for candidate in (
                "batter",
                "bowler",
                "player",
            ):

                actual_column = player_column_map.get(
                    candidate
                )

                if actual_column is not None:

                    value = row[
                        actual_column
                    ]

                    if not is_missing_value(
                        value
                    ):
                        player = str(value)
                        break

        if not player:
            player = "Unknown"

        value = get_metric_value(
            row,
            metric,
        )

        formatted_value = format_number(
            value,
            metric,
        )

        lines.append(
            f"| {rank} | {player} | "
            f"{formatted_value} |"
        )

    return (
        f"{title}\n\n"
        + "\n".join(lines)
    )


# ============================================================
# FAST PATH RESPONSE
# ============================================================

def format_fast_path_response(
    question: str,
    result: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> str:
    """
    Deterministically format fast-path query results.

    Fast-path results should not be sent back to the LLM.
    """

    response_type = get_response_type(
        question
    )

    if response_type == "ranking":

        return format_ranking_response(
            question,
            result,
            metrics,
        )

    if response_type == "comparison":

        return format_comparison_response(
            result,
            metrics,
        )

    return format_simple_response(
        question,
        result,
        metrics,
    )


# ============================================================
# LLM RESPONSE PROMPT
# ============================================================

RESPONSE_PROMPT = """
You are the response generation layer for IPL Copilot.

The SQL query has already been executed successfully.

Your job is ONLY to explain the returned result clearly.

IMPORTANT RULES:

1. Never invent statistics.

2. Never change numbers.

3. Use exactly the values present in the result.

4. Keep the answer concise.

5. Use Markdown.

6. Use a table ONLY when:
   - the question is a comparison
   - the question is a ranking

7. For a single-player statistic:
   present the result directly.

8. For multiple metrics for one player:
   use labeled lines, not a table.

9. Do not mention SQL.

10. Do not mention internal implementation.

11. Do not say "according to the database".

12. Do not add unnecessary explanations.

USER QUESTION:
{question}

RESULT:
{result}

Return only the final answer.
"""


# ============================================================
# LLM GENERATION
# ============================================================

def generate_llm_response(
    question: str,
    result: Any,
) -> str:
    """
    Use Groq for non-deterministic/non-tabular responses.
    """

    prompt = RESPONSE_PROMPT.format(
        question=question,
        result=result,
    )

    response = llm.invoke(
        prompt
    )

    content = getattr(
        response,
        "content",
        None,
    )

    if content is None:
        return str(response)

    return str(content).strip()


# ============================================================
# NORMALIZE RESULT
# ============================================================

def normalize_result(
    result: Any,
) -> Any:
    """
    Normalize possible executor outputs into a
    predictable representation.
    """

    if result is None:
        return None

    if isinstance(
        result,
        pd.DataFrame,
    ):
        return result

    if isinstance(
        result,
        pd.Series,
    ):
        return pd.DataFrame(
            [result]
        )

    if isinstance(
        result,
        dict,
    ):
        return pd.DataFrame(
            [result]
        )

    if isinstance(
        result,
        list,
    ):
        if not result:
            return pd.DataFrame()

        if all(
            isinstance(item, dict)
            for item in result
        ):
            return pd.DataFrame(
                result
            )

        return result

    return result


# ============================================================
# MAIN RESPONSE GENERATOR
# ============================================================

def generate_response(
    question: str,
    result: Any,
    use_llm: bool = True,
    metrics: Optional[list[str]] = None,
) -> str:
    """
    Main public response-generation function.

    Parameters:
        question:
            Original/resolved natural-language question.

        result:
            SQL executor result.

        use_llm:
            Whether LLM generation is allowed.

        metrics:
            Optional metrics detected by planner.
    """

    normalized_result = normalize_result(
        result
    )

    # --------------------------------------------------------
    # No result
    # --------------------------------------------------------

    if normalized_result is None:

        return "No data was found."

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if isinstance(
        normalized_result,
        pd.DataFrame,
    ):

        if normalized_result.empty:

            return "No matching data was found."

        # Fast deterministic formatter.
        #
        # This avoids sending huge ranking tables to Groq
        # and prevents token-limit errors.
        return format_fast_path_response(
            question,
            normalized_result,
            metrics,
        )

    # --------------------------------------------------------
    # Scalar
    # --------------------------------------------------------

    if isinstance(
        normalized_result,
        (int, float),
    ):

        metrics = metrics or detect_metrics_from_question(
            question
        )

        metric = (
            metrics[0]
            if metrics
            else None
        )

        formatted = format_number(
            normalized_result,
            metric,
        )

        if metric:

            unit = METRIC_UNITS.get(
                metric,
                "",
            )

            label = METRIC_LABELS.get(
                metric,
                metric.title(),
            )

            if unit:
                return (
                    f"### {label}\n\n"
                    f"**{formatted} {unit}**"
                )

            return (
                f"### {label}\n\n"
                f"**{formatted}**"
            )

        return (
            f"**{formatted}**"
        )

    # --------------------------------------------------------
    # Other structured result
    # --------------------------------------------------------

    if not use_llm:

        return str(
            normalized_result
        )

    return generate_llm_response(
        question,
        normalized_result,
    )


# ============================================================
# CLI TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "IPL Copilot - Response Generator"
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        if not question:
            continue

        print(
            "\nThis module expects an "
            "already executed SQL result."
        )