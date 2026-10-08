from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from .entity_resolver import (
    players,
    teams,
    TEAM_ALIASES,
    resolve_entity,
    find_player_mentions,
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
    max_tokens=900,
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
        description="A single safe read-only SQL query.",
    )

    reason: Optional[str] = Field(
        default=None,
        description="Reason when the request is unsupported.",
    )


structured_llm = llm.with_structured_output(SQLPlan)


# ============================================================
# METRIC ROLES
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


# ============================================================
# METRIC PHRASES
# ============================================================

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


# ============================================================
# LEGAL BALLS
# ============================================================

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


# ============================================================
# AUTHORITATIVE METRIC SQL
# ============================================================

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
    return (
        " ".join(
            str(value)
            .lower()
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )
    )


# ============================================================
# METRIC DETECTION
# ============================================================

def detect_requested_metrics(
    question: str,
) -> list[str]:

    normalized_question = normalize(
        question
    )

    matches = []

    for metric, phrases in METRIC_PHRASES.items():

        for phrase in phrases:

            normalized_phrase = normalize(
                phrase
            )

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

    # Prefer longer phrase matches.
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

        occupied.append(
            (start, end)
        )

        if metric not in detected:
            detected.append(metric)

    # "partnership" already represents runs.
    if (
        "partnership_runs" in detected
        and "runs" in detected
    ):
        detected.remove("runs")

    return detected


# ============================================================
# DEFAULT RANKING METRIC
# ============================================================

def infer_default_ranking_metric(
    question: str,
) -> list[str]:
    """
    Infer a default metric when a ranking question
    specifies a batting/bowling role but no metric.

    Examples:
        Top 5 batters -> runs
        Top 5 batsmen -> runs
        Top 5 bowlers -> wickets
    """

    normalized = normalize(
        question
    )

    # Must be a ranking-style request.
    if not re.search(
        r"\b(top|highest|most|leading|rank|ranking)\b",
        normalized,
    ):
        return []

    # --------------------------------------------------------
    # Batting ranking
    # --------------------------------------------------------

    if re.search(
        r"\b(batter|batters|batsman|batsmen)\b",
        normalized,
    ):
        return ["runs"]

    # --------------------------------------------------------
    # Bowling ranking
    # --------------------------------------------------------

    if re.search(
        r"\b(bowler|bowlers)\b",
        normalized,
    ):
        return ["wickets"]

    return []


# ============================================================
# ENTITY DETECTION
# ============================================================
def detect_entity_mentions(question: str):
    """
    Detect players and teams using the dynamic entity resolver.

    Player names/aliases come from the dataset through
    entity_resolver.py. No individual player names are hardcoded.
    """

    normalized_question = normalize(question)

    mentions = []

    # ========================================================
    # PLAYERS
    # ========================================================

    player_mentions = find_player_mentions(
        question
    )

    for mention in player_mentions:

        mentions.append(
            {
                "type": "player",
                "value": mention["value"],
            }
        )

    # ========================================================
    # TEAMS
    # ========================================================

    for team in teams:

        normalized_team = normalize(
            team
        )

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

    # ========================================================
    # TEAM ALIASES
    # ========================================================

    for alias, canonical_team in TEAM_ALIASES.items():

        normalized_alias = normalize(
            alias
        )

        if re.search(
            rf"\b{re.escape(normalized_alias)}\b",
            normalized_question,
        ):

            mentions.append(
                {
                    "type": "team",
                    "value": canonical_team,
                }
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_mentions = []

    seen = set()

    for mention in mentions:

        key = (
            mention["type"],
            normalize(
                mention["value"]
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_mentions.append(
            mention
        )

    return unique_mentions


# ============================================================
# ENTITY RESOLUTION
# ============================================================

def resolve_question_entities(
    question: str,
):

    mentions = detect_entity_mentions(
        question
    )

    resolved_question = question
    resolutions = []

    for mention in mentions:

        entity_type = mention["type"]
        value = mention["value"]

        result = resolve_entity(
            value,
            entity_type,
        )

        status = result.get(
            "status"
        )

        if status == "resolved":

            resolved_value = result[
                "resolved_value"
            ]

            pattern = re.compile(
                re.escape(value),
                re.IGNORECASE,
            )

            resolved_question = (
                pattern.sub(
                    resolved_value,
                    resolved_question,
                    count=1,
                )
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

    return (
        resolved_question,
        resolutions,
    )


# ============================================================
# SCHEMA CONTEXT
# ============================================================

def build_schema_context(
    schema,
) -> str:

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

        if isinstance(
            column,
            dict,
        ):

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

    return "\n".join(
        lines
    )


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

    return "\n".join(
        lines
    )


# ============================================================
# LANGCHAIN PROMPT
# ============================================================

PLANNER_PROMPT = """
You are the SQL planning engine for IPL Copilot.

Generate ONE safe DuckDB SQL query for the user's IPL analytics question.

DATABASE TABLE:
{table_name}

SCHEMA:
{schema}

REQUESTED METRICS:
{metrics}

USER QUESTION:
{question}

IMPORTANT RULES
==============

1. Return only a single read-only SELECT or WITH query.

2. Never use:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   CREATE
   TRUNCATE
   REPLACE
   MERGE
   COPY
   EXPORT
   IMPORT
   ATTACH
   DETACH

3. Never use external file or network functions.

4. Use only the supplied table and schema.

5. Quote hyphenated columns.
   Example:
   "non-striker"

6. Every requested registered metric MUST use its exact placeholder.

   Example:
   __METRIC_runs__ AS total_runs

7. A metric placeholder represents the COMPLETE metric expression.

   Never wrap it inside:
   SUM()
   AVG()
   COUNT()
   MIN()
   MAX()

8. Always use simple aliases.

   Examples:
   __METRIC_runs__ AS total_runs
   __METRIC_fours__ AS total_fours
   __METRIC_sixes__ AS total_sixes
   __METRIC_wickets__ AS total_wickets
   __METRIC_strike_rate__ AS strike_rate

9. For batting player statistics:
   batter = 'PLAYER'

10. For bowling player statistics:
    bowler = 'PLAYER'

11. For team batting statistics:
    BattingTeam = 'TEAM'

12. For against/vs/versus TEAM:
    TEAM is the opponent.

    Identify matches using:

    ID IN (
        SELECT DISTINCT ID
        FROM {table_name}
        WHERE BattingTeam = 'TEAM'
    )

    Then exclude the opponent innings when appropriate.

13. Partnership questions must match both directions:

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

14. For batting rankings:
    GROUP BY batter

15. For bowling rankings:
    GROUP BY bowler

16. Order rankings by the requested metric.

17. Do not add unnecessary GROUP BY columns.

18. Do not invent tables, columns, players, teams, or metrics.

19. Return unsupported when the dataset cannot answer the question.
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

    result = structured_llm.invoke(
        prompt
    )

    if not isinstance(
        result,
        SQLPlan,
    ):

        result = SQLPlan.model_validate(
            result
        )

    return result


# ============================================================
# METRIC PLACEHOLDER REPLACEMENT
# ============================================================

def replace_metric_placeholders(
    sql: str,
) -> str:

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

        pattern = re.compile(
            rf"""
            \b(?:SUM|AVG|COUNT|MIN|MAX)
            \s*\(
                \s*{re.escape(placeholder)}\s*
            \)
            """,
            flags=re.IGNORECASE
            | re.VERBOSE,
        )

        result = pattern.sub(
            placeholder,
            result,
        )

    return result


# ============================================================
# METRIC EXPRESSION ENFORCEMENT
# ============================================================

def enforce_metric_placeholders(
    sql: str,
    requested_metrics: list[str],
) -> str:
    """
    Replace common LLM-generated metric expressions
    with authoritative placeholders.
    """

    result = sql

    simple_patterns = {
        "runs": [
            r"""
            SUM\s*\(
                \s*(?:batsman_run|total_run)\s*
            \)
            """,
        ],

        "fours": [
            r"""
            SUM\s*\(
                \s*CASE\s+WHEN\s+batsman_run\s*=\s*4
                .*?
                END\s*
            \)
            """,
        ],

        "sixes": [
            r"""
            SUM\s*\(
                \s*CASE\s+WHEN\s+batsman_run\s*=\s*6
                .*?
                END\s*
            \)
            """,
        ],
    }

    for metric in requested_metrics:

        placeholder = (
            f"__METRIC_{metric}__"
        )

        if placeholder in result:
            continue

        for pattern in simple_patterns.get(
            metric,
            [],
        ):

            result = re.sub(
                pattern,
                placeholder,
                result,
                flags=re.IGNORECASE
                | re.DOTALL
                | re.VERBOSE,
            )

            if placeholder in result:
                break

    return result


# ============================================================
# METRIC ALIAS REPAIR
# ============================================================

def repair_metric_aliases(
    sql: str,
) -> str:
    """
    Repair invalid aliases after metric placeholders.
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

        placeholder = (
            f"__METRIC_{metric}__"
        )

        pattern = re.compile(
            rf"""
            ({re.escape(placeholder)})
            \s+AS\s+
            (?:
                SUM|
                AVG|
                COUNT|
                MIN|
                MAX
            )
            \s*\(
                .*?
            \)
            """,
            flags=re.IGNORECASE
            | re.DOTALL
            | re.VERBOSE,
        )

        result = pattern.sub(
            rf"\1 AS {alias}",
            result,
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

    result = sql

    for column in columns:

        if isinstance(
            column,
            dict,
        ):

            actual_name = column.get(
                "name",
                "",
            )

        else:

            actual_name = str(
                column
            )

        if not actual_name:
            continue

        # Only repair names that contain
        # spaces, hyphens or other special chars.
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

            result = re.sub(
                rf"\b{re.escape(normalized_variant)}\b",
                f'"{actual_name}"',
                result,
            )

    return result


# ============================================================
# SQL SAFETY VALIDATION
# ============================================================

def validate_sql(
    sql: str,
) -> str:

    if not isinstance(
        sql,
        str,
    ):

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
# SQL BALANCE VALIDATION
# ============================================================

def validate_sql_balance(
    sql: str,
) -> str:
    """
    Catch incomplete SQL caused by truncated output.
    """

    if not isinstance(
        sql,
        str,
    ):

        raise ValueError(
            "Generated SQL must be a string."
        )

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
# METRIC PLACEHOLDER VALIDATION
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
# FAST PATH
# ============================================================

def try_fast_path(
    question: str,
    resolved_question: str,
    resolutions: list,
    requested_metrics: list[str],
):
    """
    Handle common IPL analytics queries without calling Groq
    for SQL planning.
    """

    if not requested_metrics:
        return None

    normalized = normalize(
        resolved_question
    )

    player_resolutions = [
        item["resolved"]
        for item in resolutions
        if item["type"] == "player"
    ]

    team_resolutions = [
        item["resolved"]
        for item in resolutions
        if item["type"] == "team"
    ]

    opponent_phrases = [
        "against",
        "vs",
        "versus",
    ]

    has_opponent_phrase = any(
        phrase in normalized
        for phrase in opponent_phrases
    )

    # ========================================================
    # PARTNERSHIP
    # ========================================================

    if (
        requested_metrics
        == ["partnership_runs"]
        and len(player_resolutions) == 2
    ):

        player_a = player_resolutions[0]
        player_b = player_resolutions[1]

        return f"""
SELECT
    __METRIC_partnership_runs__ AS partnership_runs
FROM {TABLE_NAME}
WHERE
    (
        batter = '{player_a}'
        AND "non-striker" = '{player_b}'
    )
    OR
    (
        batter = '{player_b}'
        AND "non-striker" = '{player_a}'
    )
""".strip()

    # ========================================================
    # RANKING LIMIT
    # ========================================================

    ranking_limit = None

    top_match = re.search(
        r"\btop\s+(\d+)\b",
        normalized,
    )

    if top_match:

        ranking_limit = int(
            top_match.group(1)
        )

    elif re.search(
        r"\b(most|highest|maximum|max|leading)\b",
        normalized,
    ):

        ranking_limit = 1

    # ========================================================
    # RANKINGS
    # ========================================================

    if (
        ranking_limit is not None
        and 1 <= ranking_limit <= 100
        and len(player_resolutions) == 0
        and len(team_resolutions) == 0
    ):

        ranking_metric = None

        ranking_candidates = [
            "runs",
            "fours",
            "sixes",
            "boundaries",
            "balls_faced",
            "strike_rate",
            "dismissals",
            "batting_average",
            "wickets",
            "balls_bowled",
            "runs_conceded",
            "economy",
            "bowling_average",
            "dot_balls",
        ]

        for metric in ranking_candidates:

            if metric in requested_metrics:

                ranking_metric = metric
                break

        if ranking_metric:

            role = METRIC_ROLES.get(
                ranking_metric
            )

            # ------------------------------------------------
            # Batting ranking
            # ------------------------------------------------

            if role == "batter":

                alias = (
                    ranking_metric
                )

                return f"""
SELECT
    batter,
    __METRIC_{ranking_metric}__ AS {alias}
FROM {TABLE_NAME}
GROUP BY batter
ORDER BY {alias} DESC
LIMIT {ranking_limit}
""".strip()

            # ------------------------------------------------
            # Bowling ranking
            # ------------------------------------------------

            if role == "bowler":

                alias = (
                    ranking_metric
                )

                return f"""
SELECT
    bowler,
    __METRIC_{ranking_metric}__ AS {alias}
FROM {TABLE_NAME}
GROUP BY bowler
ORDER BY {alias} DESC
LIMIT {ranking_limit}
""".strip()

    # ========================================================
    # MOST RUNS AGAINST OPPONENT
    # ========================================================

    if (
        "runs" in requested_metrics
        and len(team_resolutions) == 1
        and len(player_resolutions) == 0
        and has_opponent_phrase
        and re.search(
            r"\b(most|highest|maximum|max)\b",
            normalized,
        )
    ):

        opponent = team_resolutions[0]

        return f"""
SELECT
    batter,
    __METRIC_runs__ AS total_runs
FROM {TABLE_NAME}
WHERE
    ID IN (
        SELECT DISTINCT ID
        FROM {TABLE_NAME}
        WHERE BattingTeam = '{opponent}'
    )
    AND BattingTeam <> '{opponent}'
GROUP BY batter
ORDER BY total_runs DESC
LIMIT 1
""".strip()

    # ========================================================
    # SINGLE PLAYER + SINGLE METRIC
    # ========================================================

    if (
        len(player_resolutions) == 1
        and len(requested_metrics) == 1
    ):

        player = player_resolutions[0]
        metric = requested_metrics[0]

        role = METRIC_ROLES.get(
            metric
        )

        # ----------------------------------------------------
        # Batter
        # ----------------------------------------------------

        if role == "batter":

            if (
                len(team_resolutions) == 1
                and has_opponent_phrase
            ):

                opponent = (
                    team_resolutions[0]
                )

                return f"""
SELECT
    __METRIC_{metric}__ AS {metric}
FROM {TABLE_NAME}
WHERE
    batter = '{player}'
    AND ID IN (
        SELECT DISTINCT ID
        FROM {TABLE_NAME}
        WHERE BattingTeam = '{opponent}'
    )
""".strip()

            return f"""
SELECT
    __METRIC_{metric}__ AS {metric}
FROM {TABLE_NAME}
WHERE batter = '{player}'
""".strip()

        # ----------------------------------------------------
        # Bowler
        # ----------------------------------------------------

        if role == "bowler":

            if (
                len(team_resolutions) == 1
                and has_opponent_phrase
            ):

                opponent = (
                    team_resolutions[0]
                )

                return f"""
SELECT
    __METRIC_{metric}__ AS {metric}
FROM {TABLE_NAME}
WHERE
    bowler = '{player}'
    AND ID IN (
        SELECT DISTINCT ID
        FROM {TABLE_NAME}
        WHERE BattingTeam = '{opponent}'
    )
""".strip()

            return f"""
SELECT
    __METRIC_{metric}__ AS {metric}
FROM {TABLE_NAME}
WHERE bowler = '{player}'
""".strip()

    # ========================================================
    # NO FAST PATH
    # ========================================================

    return None


# ============================================================
# MAIN SQL GENERATION PIPELINE
# ============================================================

def generate_sql(
    question: str,
):

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():

        return {
            "status": "unsupported",
            "sql": None,
            "resolved_question": question,
            "reason": "Question is empty.",
            "resolutions": [],
            "metrics": [],
            "fast_path": False,
        }

    # --------------------------------------------------------
    # Resolve entities
    # --------------------------------------------------------

    (
        resolved_question,
        resolutions,
    ) = resolve_question_entities(
        question
    )

    # --------------------------------------------------------
    # Inspect dataset schema
    # --------------------------------------------------------

    schema = inspect_csv(
        DATASET_PATH
    )

    schema_context = (
        build_schema_context(
            schema
        )
    )

    # --------------------------------------------------------
    # Detect metrics
    # --------------------------------------------------------

    requested_metrics = (
        detect_requested_metrics(
            resolved_question
        )
    )

    # --------------------------------------------------------
    # Infer default ranking metrics
    #
    # Top 5 batters -> runs
    # Top 5 bowlers -> wickets
    # --------------------------------------------------------

    if not requested_metrics:

        requested_metrics = (
            infer_default_ranking_metric(
                resolved_question
            )
        )

    metric_context = (
        build_metric_context(
            requested_metrics
        )
    )

    # ========================================================
    # FAST PATH FIRST
    # ========================================================

    fast_sql = try_fast_path(
        question,
        resolved_question,
        resolutions,
        requested_metrics,
    )

    fast_path = (
        fast_sql is not None
    )

    # ========================================================
    # USE FAST PATH OR GROQ
    # ========================================================

    if fast_path:

        sql = fast_sql

    else:

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
                "resolutions": resolutions,
                "metrics": requested_metrics,
                "fast_path": False,
            }

        sql = result.sql

        if not sql:

            raise ValueError(
                "Planner did not return SQL."
            )

    # ========================================================
    # GENERIC REPAIRS
    # ========================================================

    sql = normalize_metric_placeholder_wrappers(
        sql
    )

    sql = enforce_metric_placeholders(
        sql,
        requested_metrics,
    )

    sql = repair_metric_aliases(
        sql
    )

    sql = repair_schema_column_names(
        sql,
        schema,
    )

    # ========================================================
    # VALIDATE METRICS
    # ========================================================

    missing_metrics = (
        validate_metric_placeholders(
            sql,
            requested_metrics,
        )
    )

    if missing_metrics:

        raise ValueError(
            "Required metric placeholders are missing: "
            + ", ".join(
                missing_metrics
            )
        )

    # ========================================================
    # VALIDATE SQL BEFORE REPLACEMENT
    # ========================================================

    sql = validate_sql(
        sql
    )

    sql = validate_sql_balance(
        sql
    )

    # ========================================================
    # REPLACE AUTHORITATIVE METRICS
    # ========================================================

    sql = replace_metric_placeholders(
        sql
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    sql = validate_sql_balance(
        sql
    )

    sql = validate_sql(
        sql
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "status": "ready",
        "sql": sql,
        "resolved_question": resolved_question,
        "reason": None,
        "resolutions": resolutions,
        "metrics": requested_metrics,
        "fast_path": fast_path,
    }


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