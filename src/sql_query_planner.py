import json
import os
import re
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from .entity_resolver import (
    players,
    teams,
    TEAM_ALIASES,
    resolve_entity,
)
from .schema_inspector import inspect_csv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

MODEL = os.getenv("GROQ_MODEL")

if not MODEL:
    raise ValueError("GROQ_MODEL is missing from .env")

DATASET_PATH = "data/ipl_ball_by_ball_cleaned.csv"
TABLE_NAME = "ipl_ball_by_ball_cleaned"


# ============================================================
# LANGCHAIN GROQ
# ============================================================

llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=2000,
    reasoning_effort="none",
)
# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class SQLPlan(BaseModel):
    status: Literal["ready", "unsupported"] = Field(
        description="Whether the question can be answered using the IPL dataset."
    )

    sql: Optional[str] = Field(
        default=None,
        description="A single safe read-only SQL query."
    )

    reason: Optional[str] = Field(
        default=None,
        description="Reason when the request is unsupported."
    )


structured_llm = llm.with_structured_output(SQLPlan)


# ============================================================
# METRICS
# ============================================================

METRIC_ROLES = {
    "runs": "batter",
    "fours": "batter",
    "sixes": "batter",
    "boundaries": "batter",
    "balls_faced": "batter",
    "strike_rate": "batter",
    "dismissals": "batter",
    "batting_average": "batter",

    "wickets": "bowler",
    "balls_bowled": "bowler",
    "runs_conceded": "bowler",
    "economy": "bowler",
    "bowling_average": "bowler",
    "dot_balls": "bowler",

    "partnership_runs": "player_pair",
}


METRIC_PHRASES = {
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
        "six",
    ],

    "fours": [
        "fours",
        "four",
    ],

    "wickets": [
        "wickets",
        "wicket",
    ],

    "runs": [
        "runs",
        "run",
    ],

    "economy": [
        "economy rate",
        "economy",
    ],

    "partnership_runs": [
        "partnership runs",
        "partnership",
    ],
}


# ============================================================
# SQL METRIC EXPRESSIONS
# ============================================================

WIDE_VALUES = "'wide', 'wides'"

NO_BALL_VALUES = (
    "'no ball', "
    "'no-ball', "
    "'noball', "
    "'noballs', "
    "'no balls'"
)

NON_BOWLER_WICKET_VALUES = (
    "'run out', "
    "'retired hurt', "
    "'retired out', "
    "'obstructing the field'"
)

BYE_VALUES = "'bye', 'byes'"

LEG_BYE_VALUES = (
    "'leg bye', "
    "'leg-bye', "
    "'legbye', "
    "'legbyes', "
    "'leg byes'"
)

PENALTY_VALUES = "'penalty', 'penalty runs'"


LEGAL_BALLS = f"""
SUM(
    CASE
        WHEN COALESCE(
            LOWER(TRIM(CAST(extra_type AS VARCHAR))),
            ''
        ) NOT IN (
            {WIDE_VALUES},
            {NO_BALL_VALUES}
        )
        THEN 1
        ELSE 0
    END
)
"""


METRIC_SQL_EXPRESSIONS = {
    "runs": """
SUM(batsman_run)
""",

    "fours": """
SUM(
    CASE
        WHEN batsman_run = 4
         AND LOWER(
             TRIM(CAST(non_boundary AS VARCHAR))
         ) NOT IN (
             '1',
             'true',
             'yes',
             'y'
         )
        THEN 1
        ELSE 0
    END
)
""",

    "sixes": """
SUM(
    CASE
        WHEN batsman_run = 6
         AND LOWER(
             TRIM(CAST(non_boundary AS VARCHAR))
         ) NOT IN (
             '1',
             'true',
             'yes',
             'y'
         )
        THEN 1
        ELSE 0
    END
)
""",

    "boundaries": """
(
    SUM(
        CASE
            WHEN batsman_run = 4
             AND LOWER(
                 TRIM(CAST(non_boundary AS VARCHAR))
             ) NOT IN (
                 '1',
                 'true',
                 'yes',
                 'y'
             )
            THEN 1
            ELSE 0
        END
    )
    +
    SUM(
        CASE
            WHEN batsman_run = 6
             AND LOWER(
                 TRIM(CAST(non_boundary AS VARCHAR))
             ) NOT IN (
                 '1',
                 'true',
                 'yes',
                 'y'
             )
            THEN 1
            ELSE 0
        END
    )
)
""",

    "balls_faced": LEGAL_BALLS,

    "balls_bowled": LEGAL_BALLS,

    "strike_rate": f"""
(
    SUM(batsman_run) * 100.0
    /
    NULLIF(
        {LEGAL_BALLS},
        0
    )
)
""",

    "dismissals": """
SUM(
    CASE
        WHEN LOWER(
            TRIM(CAST(player_out AS VARCHAR))
        ) NOT IN (
            '',
            'none',
            'nan',
            'null'
        )
        THEN 1
        ELSE 0
    END
)
""",

    "wickets": f"""
SUM(
    CASE
        WHEN LOWER(
            TRIM(CAST(isWicketDelivery AS VARCHAR))
        ) IN (
            '1',
            'true',
            'yes',
            'y'
        )
        AND LOWER(
            TRIM(CAST(kind AS VARCHAR))
        ) NOT IN (
            {NON_BOWLER_WICKET_VALUES}
        )
        THEN 1
        ELSE 0
    END
)
""",

    "runs_conceded": f"""
SUM(
    CASE
        WHEN LOWER(
            TRIM(CAST(extra_type AS VARCHAR))
        ) IN (
            {BYE_VALUES},
            {LEG_BYE_VALUES},
            {PENALTY_VALUES}
        )
        THEN batsman_run
        ELSE total_run
    END
)
""",

    "economy": f"""
(
    SUM(
        CASE
            WHEN LOWER(
                TRIM(CAST(extra_type AS VARCHAR))
            ) IN (
                {BYE_VALUES},
                {LEG_BYE_VALUES},
                {PENALTY_VALUES}
            )
            THEN batsman_run
            ELSE total_run
        END
    )
    /
    NULLIF(
        {LEGAL_BALLS},
        0
    )
    * 6
)
""",

    "batting_average": """
(
    SUM(batsman_run)
    /
    NULLIF(
        SUM(
            CASE
                WHEN LOWER(
                    TRIM(CAST(player_out AS VARCHAR))
                ) NOT IN (
                    '',
                    'none',
                    'nan',
                    'null'
                )
                THEN 1
                ELSE 0
            END
        ),
        0
    )
)
""",

    "bowling_average": f"""
(
    SUM(
        CASE
            WHEN LOWER(
                TRIM(CAST(extra_type AS VARCHAR))
            ) IN (
                {BYE_VALUES},
                {LEG_BYE_VALUES},
                {PENALTY_VALUES}
            )
            THEN batsman_run
            ELSE total_run
        END
    )
    /
    NULLIF(
        SUM(
            CASE
                WHEN LOWER(
                    TRIM(CAST(isWicketDelivery AS VARCHAR))
                ) IN (
                    '1',
                    'true',
                    'yes',
                    'y'
                )
                AND LOWER(
                    TRIM(CAST(kind AS VARCHAR))
                ) NOT IN (
                    {NON_BOWLER_WICKET_VALUES}
                )
                THEN 1
                ELSE 0
            END
        ),
        0
    )
)
""",

    "partnership_runs": """
SUM(total_run)
""",

    "dot_balls": f"""
SUM(
    CASE
        WHEN batsman_run = 0
         AND COALESCE(
             LOWER(
                 TRIM(CAST(extra_type AS VARCHAR))
             ),
             ''
         ) NOT IN (
             {WIDE_VALUES},
             {NO_BALL_VALUES}
         )
        THEN 1
        ELSE 0
    END
)
""",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value: str) -> str:
    return " ".join(
        str(value)
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )


# ============================================================
# METRIC DETECTION
# ============================================================

def detect_requested_metrics(question: str) -> list[str]:
    normalized_question = normalize(question)

    matches = []

    for metric, phrases in METRIC_PHRASES.items():
        for phrase in phrases:
            normalized_phrase = normalize(phrase)

            match = re.search(
                rf"\b{re.escape(normalized_phrase)}\b",
                normalized_question,
            )

            if match:
                matches.append(
                    (
                        match.start(),
                        match.end(),
                        metric,
                    )
                )

    matches.sort(
        key=lambda item: (
            item[1] - item[0],
            -item[0],
        ),
        reverse=True,
    )

    detected = []
    occupied = []

    for start, end, metric in matches:
        overlaps = any(
            start < other_end
            and end > other_start
            for other_start, other_end in occupied
        )

        if overlaps:
            continue

        occupied.append((start, end))

        if metric not in detected:
            detected.append(metric)

    if (
        "partnership_runs" in detected
        and "runs" in detected
    ):
        detected.remove("runs")

    return detected


# ============================================================
# ENTITY DETECTION
# ============================================================

def detect_entity_mentions(question: str):
    normalized_question = normalize(question)

    mentions = []

    words = re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?",
        question,
    )

    clean_words = [
        re.sub(
            r"['’]s$",
            "",
            word,
            flags=re.IGNORECASE,
        )
        for word in words
    ]

    # --------------------------------------------------------
    # Players
    # --------------------------------------------------------

    player_surnames = set()

    for player in players:
        parts = normalize(player).split()

        if len(parts) >= 2:
            surname = parts[-1]

            if len(surname) >= 4:
                player_surnames.add(surname)

    for index, word in enumerate(clean_words):
        normalized_word = normalize(word)

        if normalized_word not in player_surnames:
            continue

        candidates = []

        for size in range(
            min(3, index + 1),
            0,
            -1,
        ):
            start = index - size + 1

            candidate = " ".join(
                clean_words[start:index + 1]
            ).strip()

            if not candidate:
                continue

            result = resolve_entity(
                candidate,
                "player",
            )

            if result.get("status") == "resolved":
                candidates.append(
                    {
                        "value": candidate,
                        "score": result.get(
                            "score",
                            0,
                        ),
                        "length": size,
                    }
                )

        if candidates:
            best = max(
                candidates,
                key=lambda item: (
                    item["score"],
                    item["length"],
                ),
            )

            mentions.append(
                {
                    "type": "player",
                    "value": best["value"],
                }
            )
        else:
            mentions.append(
                {
                    "type": "player",
                    "value": normalized_word,
                }
            )

    # --------------------------------------------------------
    # Teams
    # --------------------------------------------------------

    for team in teams:
        normalized_team = normalize(team)

        if re.search(
            rf"\b{re.escape(normalized_team)}\b",
            normalized_question,
        ):
            mentions.append(
                {
                    "type": "team",
                    "value": team,
                }
            )

    for alias in TEAM_ALIASES:
        normalized_alias = normalize(alias)

        if re.search(
            rf"\b{re.escape(normalized_alias)}\b",
            normalized_question,
        ):
            mentions.append(
                {
                    "type": "team",
                    "value": alias,
                }
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_mentions = []
    seen = set()

    for mention in mentions:
        key = (
            mention["type"],
            normalize(mention["value"]),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_mentions.append(mention)

    return unique_mentions


# ============================================================
# ENTITY RESOLUTION
# ============================================================

def resolve_question_entities(question: str):
    mentions = detect_entity_mentions(question)

    resolved_question = question
    resolutions = []

    for mention in mentions:
        entity_type = mention["type"]
        value = mention["value"]

        result = resolve_entity(
            value,
            entity_type,
        )

        status = result.get("status")

        if status == "resolved":
            resolved_value = result["resolved_value"]

            pattern = re.compile(
                re.escape(value),
                re.IGNORECASE,
            )

            resolved_question = pattern.sub(
                resolved_value,
                resolved_question,
                count=1,
            )

            resolutions.append(
                {
                    "type": entity_type,
                    "original": value,
                    "resolved": resolved_value,
                }
            )

        elif status == "ambiguous":
            candidates = result.get(
                "candidates",
                [],
            )

            raise ValueError(
                f"Ambiguous {entity_type} "
                f"'{value}'. Candidates: "
                + ", ".join(
                    str(candidate)
                    for candidate in candidates
                )
            )

        elif status == "not_found":
            raise ValueError(
                f"Could not resolve "
                f"{entity_type} '{value}'."
            )

    return resolved_question, resolutions


# ============================================================
# SCHEMA
# ============================================================

def build_schema_context(schema) -> str:
    columns = schema.get(
        "columns",
        [],
    )

    lines = [
        f"TABLE: {TABLE_NAME}",
        "",
        "COLUMNS:",
    ]

    for column in columns:
        if isinstance(column, dict):
            name = column.get(
                "name",
                "",
            )

            dtype = column.get(
                "dtype",
                "unknown",
            )

            lines.append(
                f"- {name} ({dtype})"
            )
        else:
            lines.append(
                f"- {column}"
            )

    return "\n".join(lines)


# ============================================================
# METRIC CONTEXT
# ============================================================

def build_metric_context(
    requested_metrics: list[str],
) -> str:
    lines = [
        "AVAILABLE METRIC PLACEHOLDERS:"
    ]

    for metric in requested_metrics:
        if metric not in METRIC_SQL_EXPRESSIONS:
            continue

        role = METRIC_ROLES.get(
            metric,
            "unknown",
        )

        lines.append(
            f"- __METRIC_{metric}__ "
            f"(role: {role})"
        )

    return "\n".join(lines)


# ============================================================
# LANGCHAIN PROMPT
# ============================================================

PLANNER_PROMPT = """
You are the SQL planning engine for IPL Copilot.

Convert the user's natural-language IPL analytics question
into ONE safe DuckDB SQL query.

DATABASE TABLE:
{table_name}

SCHEMA:
{schema}

REQUESTED METRICS:
{metrics}

USER QUESTION:
{question}

IMPORTANT RULES
================

1. Return only a read-only SELECT or WITH query.

2. Never use:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   CREATE
   TRUNCATE
   COPY
   EXPORT
   IMPORT
   ATTACH
   DETACH

3. Never use external file/network functions.

4. Use only columns present in the supplied schema.

5. For columns containing hyphens, use double quotes.
   Example:
   "non-striker"

6. Registered metrics MUST use their placeholders.

   Example:
   SELECT __METRIC_runs__ AS total_runs

7. A metric placeholder represents the COMPLETE metric expression.

   NEVER wrap it inside another SQL function.

   WRONG:
   SELECT SUM(__METRIC_runs__)

   WRONG:
   SELECT __METRIC_runs__ AS SUM(batsman_run)

   WRONG:
   SELECT AVG(__METRIC_runs__)

   CORRECT:
   SELECT __METRIC_runs__ AS total_runs

8. Always use a simple identifier as the alias.

   Examples:
   __METRIC_runs__ AS total_runs
   __METRIC_fours__ AS total_fours
   __METRIC_sixes__ AS total_sixes
   __METRIC_wickets__ AS total_wickets
   __METRIC_strike_rate__ AS strike_rate

9. For batting:
   use batter = 'PLAYER'

10. For bowling:
    use bowler = 'PLAYER'

11. For team batting statistics:
    use:
    BattingTeam = 'TEAM'

12. For "against", "vs", or "versus":

    The mentioned team is the opponent.

    Do NOT use that team as BattingTeam for the player.

    Instead identify matches using:

    ID IN (
        SELECT DISTINCT ID
        FROM {table_name}
        WHERE BattingTeam = 'OPPONENT'
    )

13. Partnership questions must match BOTH orientations:

    (
        batter = 'PLAYER_A'
        AND "non-striker" = 'PLAYER_B'
    )
    OR
    (
        batter = 'PLAYER_B'
        AND "non-striker" = 'PLAYER_A'
    )

    Use:
    __METRIC_partnership_runs__

14. For rankings:

    Batting:
    GROUP BY batter

    Bowling:
    GROUP BY bowler

    Order by the requested metric.

15. Do not add unnecessary GROUP BY columns.

16. Do not invent tables, columns, players or teams.

17. Return a single SQL statement.
"""


# ============================================================
# LANGCHAIN SQL PLANNER
# ============================================================

def call_planner(
    question: str,
    schema_context: str,
    metric_context: str,
) -> SQLPlan:

    prompt = PLANNER_PROMPT.format(
        table_name=TABLE_NAME,
        schema=schema_context,
        metrics=metric_context,
        question=question,
    )

    result = structured_llm.invoke(prompt)

    if not isinstance(result, SQLPlan):
        result = SQLPlan.model_validate(result)

    return result


# ============================================================
# PLACEHOLDER REPLACEMENT
# ============================================================

def replace_metric_placeholders(sql: str) -> str:
    result = sql

    for metric_name, expression in (
        METRIC_SQL_EXPRESSIONS.items()
    ):
        placeholder = (
            f"__METRIC_{metric_name}__"
        )

        result = result.replace(
            placeholder,
            expression.strip(),
        )

    return result


# ============================================================
# PLACEHOLDER WRAPPER REPAIR
# ============================================================

def normalize_metric_placeholder_wrappers(
    sql: str,
) -> str:

    result = sql

    for metric_name in METRIC_SQL_EXPRESSIONS:
        placeholder = (
            f"__METRIC_{metric_name}__"
        )

        for function_name in (
            "SUM",
            "AVG",
            "COUNT",
            "MIN",
            "MAX",
        ):
            pattern = (
                rf"{function_name}\s*"
                rf"\(\s*"
                rf"{re.escape(placeholder)}"
                rf"\s*\)"
            )

            result = re.sub(
                pattern,
                placeholder,
                result,
                flags=re.IGNORECASE,
            )

    return result


def enforce_metric_placeholders(
    sql: str,
    requested_metrics: list[str],
) -> str:
    """
    Replace common LLM-generated metric expressions with
    the authoritative metric placeholders.
    """

    result = sql

    replacements = {
        "runs": [
            r"SUM\s*\(\s*(?:batsman_run|total_run)\s*\)",
        ],

        "fours": [
            r"SUM\s*\(\s*CASE\s+WHEN\s+batsman_run\s*=\s*4.*?END\s*\)",
        ],

        "sixes": [
            r"SUM\s*\(\s*CASE\s+WHEN\s+batsman_run\s*=\s*6.*?END\s*\)",
        ],
    }

    for metric in requested_metrics:
        placeholder = f"__METRIC_{metric}__"

        if placeholder in result:
            continue

        for pattern in replacements.get(metric, []):
            result = re.sub(
                pattern,
                placeholder,
                result,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if placeholder in result:
                break

    return result


def repair_metric_aliases(sql: str) -> str:
    """
    Repairs invalid aliases such as:

        SELECT __METRIC_runs__ AS SUM(batsman_run)

    into:

        SELECT __METRIC_runs__ AS total_runs
    """

    aliases = {
        "runs": "total_runs",
        "fours": "total_fours",
        "sixes": "total_sixes",
        "boundaries": "total_boundaries",
        "balls_faced": "balls_faced",
        "strike_rate": "strike_rate",
        "dismissals": "dismissals",
        "batting_average": "batting_average",
        "wickets": "wickets",
        "balls_bowled": "balls_bowled",
        "runs_conceded": "runs_conceded",
        "economy": "economy",
        "bowling_average": "bowling_average",
        "dot_balls": "dot_balls",
        "partnership_runs": "partnership_runs",
    }

    result = sql

    for metric, alias in aliases.items():
        placeholder = f"__METRIC_{metric}__"

        pattern = (
            rf"({re.escape(placeholder)})"
            rf"\s+AS\s+"
            rf"(?:SUM|AVG|COUNT|MIN|MAX)"
            rf"\s*\([^)]*\)"
        )

        result = re.sub(
            pattern,
            rf"\1 AS {alias}",
            result,
            flags=re.IGNORECASE,
        )

    return result


# ============================================================
# SCHEMA COLUMN REPAIR
# ============================================================

def repair_schema_column_names(
    sql: str,
    schema,
) -> str:

    columns = schema.get(
        "columns",
        [],
    )

    for column in columns:
        if isinstance(column, dict):
            actual_name = column.get(
                "name",
                "",
            )
        else:
            actual_name = str(column)

        if not actual_name:
            continue

        if not re.search(
            r"[^A-Za-z0-9_]",
            actual_name,
        ):
            continue

        normalized_variant = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            actual_name,
        ).strip("_")

        if (
            normalized_variant
            and normalized_variant != actual_name
        ):
            sql = re.sub(
                rf"\b"
                rf"{re.escape(normalized_variant)}"
                rf"\b",
                f'"{actual_name}"',
                sql,
            )

    return sql


# ============================================================
# SQL SAFETY
# ============================================================

def validate_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise ValueError(
            "Generated SQL must be a string."
        )

    sql = sql.strip()

    if not sql:
        raise ValueError(
            "Generated SQL is empty."
        )

    statements = [
        statement.strip()
        for statement in sql.split(";")
        if statement.strip()
    ]

    if len(statements) != 1:
        raise ValueError(
            "Only one SQL statement is allowed."
        )

    statement = statements[0]

    normalized = re.sub(
        r"\s+",
        " ",
        statement.lower(),
    ).strip()

    if not (
        normalized.startswith("select ")
        or normalized.startswith("with ")
    ):
        raise ValueError(
            "Only SELECT or WITH queries are allowed."
        )

    forbidden = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "create ",
        "truncate ",
        "replace ",
        "merge ",
        "copy ",
        "export ",
        "import ",
        "attach ",
        "detach ",
    ]

    for keyword in forbidden:
        if keyword in normalized:
            raise ValueError(
                f"Forbidden SQL operation: "
                f"{keyword.strip()}"
            )

    external_functions = [
        "read_csv(",
        "read_csv_auto(",
        "read_json(",
        "read_parquet(",
        "httpfs",
        "parquet_scan(",
        "glob(",
    ]

    for function_name in external_functions:
        if function_name in normalized:
            raise ValueError(
                "External file/network access "
                "is not allowed."
            )

    return statement


# ============================================================
# METRIC VALIDATION
# ============================================================

def validate_metric_placeholders(
    sql: str,
    requested_metrics: list[str],
):
    missing = []

    for metric in requested_metrics:
        if metric not in METRIC_SQL_EXPRESSIONS:
            continue

        placeholder = (
            f"__METRIC_{metric}__"
        )

        if placeholder not in sql:
            missing.append(metric)

    return missing


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_sql(question: str):

    if not question or not question.strip():
        return {
            "status": "unsupported",
            "sql": None,
            "resolved_question": question,
            "reason": "Question is empty.",
        }

    # --------------------------------------------------------
    # Resolve entities
    # --------------------------------------------------------

    (
        resolved_question,
        resolutions,
    ) = resolve_question_entities(question)

    # --------------------------------------------------------
    # Inspect dataset
    # --------------------------------------------------------

    schema = inspect_csv(
        DATASET_PATH
    )

    schema_context = build_schema_context(
        schema
    )

    # --------------------------------------------------------
    # Detect metrics
    # --------------------------------------------------------

    requested_metrics = (
        detect_requested_metrics(
            resolved_question
        )
    )

    metric_context = build_metric_context(
        requested_metrics
    )

    # --------------------------------------------------------
    # LangChain planner
    # --------------------------------------------------------

    result = call_planner(
        resolved_question,
        schema_context,
        metric_context,
    )

    if result.status != "ready":
        return {
            "status": "unsupported",
            "sql": None,
            "resolved_question": resolved_question,
            "reason": (
                result.reason
                or "Question is unsupported."
            ),
        }

    sql = result.sql

    if not sql:
        raise ValueError(
            "Planner did not return SQL."
        )

    # --------------------------------------------------------
    # Generic repairs
    # --------------------------------------------------------

    sql = normalize_metric_placeholder_wrappers(
        sql
    )

    sql = enforce_metric_placeholders(
        sql,
        requested_metrics,
    )

    sql = repair_metric_aliases(sql)

    sql = repair_schema_column_names(
        sql,
        schema,
    )

    # --------------------------------------------------------
    # Validate requested metrics
    # --------------------------------------------------------

    missing_metrics = (
        validate_metric_placeholders(
            sql,
            requested_metrics,
        )
    )

    if missing_metrics:
        raise ValueError(
            "LLM did not use the required metric "
            "placeholders: "
            + ", ".join(missing_metrics)
        )

    # --------------------------------------------------------
    # Validate SQL before replacement
    # --------------------------------------------------------

    sql = validate_sql(sql)

    # --------------------------------------------------------
    # Replace authoritative metrics
    # --------------------------------------------------------

    sql = replace_metric_placeholders(
        sql
    )

    sql = validate_sql_balance(sql)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    sql = validate_sql(sql)

    return {
        "status": "ready",
        "sql": sql,
        "resolved_question": resolved_question,
        "reason": None,
        "resolutions": resolutions,
        "metrics": requested_metrics,
    }


def validate_sql_balance(sql):
    """
    Catch incomplete SQL caused by truncated LLM output.
    """
    if not isinstance(sql, str):
        raise ValueError("Generated SQL must be a string.")

    if sql.count("(") != sql.count(")"):
        raise ValueError(
            "Generated SQL has unbalanced parentheses."
        )

    single_quotes = 0
    double_quotes = 0

    escaped = False

    for char in sql:
        if char == "\\" and not escaped:
            escaped = True
            continue

        if char == "'" and not escaped:
            single_quotes += 1

        elif char == '"' and not escaped:
            double_quotes += 1

        escaped = False

    if single_quotes % 2 != 0:
        raise ValueError(
            "Generated SQL has an unclosed single quote."
        )

    if double_quotes % 2 != 0:
        raise ValueError(
            "Generated SQL has an unclosed double quote."
        )

    return sql

# ============================================================
# CLI TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "IPL Copilot - LangChain Groq SQL Planner"
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        if not question:
            continue

        try:

            result = generate_sql(
                question
            )

            print(
                "\nPlanner Output:"
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

        except Exception as error:

            print(
                "\nPlanner Error:"
            )

            print(error)