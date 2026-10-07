import json
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .entity_resolver import (
    players,
    teams,
    TEAM_ALIASES,
    resolve_entity,
)
from .schema_inspector import inspect_csv

load_dotenv()


# ============================================================
# ENVIRONMENT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

DATASET_PATH = "data/ipl_ball_by_ball_cleaned.csv"

TABLE_NAME = "ipl_ball_by_ball_cleaned"

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GEMINI HELPER
# ============================================================

def call_gemini(
    system_prompt,
    user_prompt,
    max_output_tokens=800,
):
    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )

    content = response.text

    if not content:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return content


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
# SQL VALUE GROUPS
# ============================================================

WIDE_VALUES = (
    "'wide', 'wides'"
)

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

BYE_VALUES = (
    "'bye', 'byes'"
)

LEG_BYE_VALUES = (
    "'leg bye', "
    "'leg-bye', "
    "'legbye', "
    "'legbyes', "
    "'leg byes'"
)

PENALTY_VALUES = (
    "'penalty', 'penalty runs'"
)


# ============================================================
# LEGAL BALLS EXPRESSION
# ============================================================

LEGAL_BALLS_EXPRESSION = f"""
SUM(
    CASE
        WHEN COALESCE(
            LOWER(
                TRIM(
                    CAST(extra_type AS VARCHAR)
                )
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
"""


# ============================================================
# AUTHORITATIVE METRIC SQL EXPRESSIONS
# ============================================================

METRIC_SQL_EXPRESSIONS = {

    # --------------------------------------------------------
    # Batting
    # --------------------------------------------------------

    "runs": """
SUM(batsman_run)
""",

    "fours": """
SUM(
    CASE
        WHEN batsman_run = 4
         AND LOWER(
             TRIM(
                 CAST(non_boundary AS VARCHAR)
             )
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
             TRIM(
                 CAST(non_boundary AS VARCHAR)
             )
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
                 TRIM(
                     CAST(non_boundary AS VARCHAR)
                 )
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
                 TRIM(
                     CAST(non_boundary AS VARCHAR)
                 )
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

    "balls_faced": LEGAL_BALLS_EXPRESSION,

    "balls_bowled": LEGAL_BALLS_EXPRESSION,

    "strike_rate": f"""
(
    SUM(batsman_run) * 100.0
    /
    NULLIF(
        {LEGAL_BALLS_EXPRESSION},
        0
    )
)
""",

    "dismissals": """
SUM(
    CASE
        WHEN LOWER(
            TRIM(
                CAST(player_out AS VARCHAR)
            )
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

    # --------------------------------------------------------
    # Bowling
    # --------------------------------------------------------

    "wickets": f"""
SUM(
    CASE
        WHEN LOWER(
            TRIM(
                CAST(isWicketDelivery AS VARCHAR)
            )
        ) IN (
            '1',
            'true',
            'yes',
            'y'
        )
        AND LOWER(
            TRIM(
                CAST(kind AS VARCHAR)
            )
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
            TRIM(
                CAST(extra_type AS VARCHAR)
            )
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
    (
        SUM(
            CASE
                WHEN LOWER(
                    TRIM(
                        CAST(extra_type AS VARCHAR)
                    )
                ) IN (
                    {BYE_VALUES},
                    {LEG_BYE_VALUES},
                    {PENALTY_VALUES}
                )
                THEN batsman_run
                ELSE total_run
            END
        )
    )
    /
    NULLIF(
        {LEGAL_BALLS_EXPRESSION},
        0
    )
    * 6
)
""",

    "batting_average": f"""
(
    SUM(batsman_run)
    /
    NULLIF(
        SUM(
            CASE
                WHEN LOWER(
                    TRIM(
                        CAST(player_out AS VARCHAR)
                    )
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
    (
        SUM(
            CASE
                WHEN LOWER(
                    TRIM(
                        CAST(extra_type AS VARCHAR)
                    )
                ) IN (
                    {BYE_VALUES},
                    {LEG_BYE_VALUES},
                    {PENALTY_VALUES}
                )
                THEN batsman_run
                ELSE total_run
            END
        )
    )
    /
    NULLIF(
        SUM(
            CASE
                WHEN LOWER(
                    TRIM(
                        CAST(isWicketDelivery AS VARCHAR)
                    )
                ) IN (
                    '1',
                    'true',
                    'yes',
                    'y'
                )
                AND LOWER(
                    TRIM(
                        CAST(kind AS VARCHAR)
                    )
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

    # --------------------------------------------------------
    # Partnership
    # --------------------------------------------------------

    "partnership_runs": """
SUM(total_run)
""",

    # --------------------------------------------------------
    # Dot balls
    # --------------------------------------------------------

    "dot_balls": f"""
SUM(
    CASE
        WHEN batsman_run = 0
         AND COALESCE(
             LOWER(
                 TRIM(
                     CAST(extra_type AS VARCHAR)
                 )
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
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the SQL planning engine for IPL Copilot.

Your job is to convert a natural-language IPL analytics question
into safe read-only SQL.

You must use the supplied dataset schema.

Never invent:

- tables
- columns
- players
- teams
- filters
- grouping
- limits
- metrics

The database contains ball-by-ball cricket data.

============================================================
METRIC RULES
============================================================

For registered cricket metrics, NEVER invent the arithmetic.

Use the supplied metric placeholder.

A placeholder has this exact format:

__METRIC_<metric_name>__

Examples:

__METRIC_runs__
__METRIC_strike_rate__
__METRIC_wickets__
__METRIC_fours__
__METRIC_sixes__
__METRIC_partnership_runs__

The application will replace the placeholders with authoritative
metric expressions.

Therefore:

DO NOT write your own formula for a registered metric.

DO NOT replace a metric with COUNT(*).

DO NOT replace a metric with AVG().

DO NOT invent a strike-rate formula.

DO NOT invent a wicket formula.

DO NOT invent boundary logic.

A metric placeholder already represents a complete aggregate
expression.

NEVER wrap a metric placeholder inside another aggregate.

WRONG:

SUM(__METRIC_runs__)

AVG(__METRIC_strike_rate__)

COUNT(__METRIC_wickets__)

CORRECT:

__METRIC_runs__

__METRIC_strike_rate__

__METRIC_wickets__

============================================================
AVAILABLE METRICS
============================================================

runs
fours
sixes
boundaries
balls_faced
strike_rate
dismissals
batting_average
wickets
balls_bowled
runs_conceded
economy
bowling_average
dot_balls
partnership_runs

============================================================
PLAYER ROLES
============================================================

runs -> batter
fours -> batter
sixes -> batter
boundaries -> batter
balls_faced -> batter
strike_rate -> batter
dismissals -> batter
batting_average -> batter

wickets -> bowler
balls_bowled -> bowler
runs_conceded -> bowler
economy -> bowler
bowling_average -> bowler
dot_balls -> bowler

partnership_runs -> player_pair

============================================================
NORMAL PLAYER QUESTIONS
============================================================

For a specific player's batting career statistic:

WHERE batter = 'player'

For a bowling statistic:

WHERE bowler = 'player'

============================================================
RANKINGS
============================================================

For a batting ranking:

GROUP BY batter

For a bowling ranking:

GROUP BY bowler

For ranking:

ORDER BY the requested metric DESC or ASC

LIMIT only when explicitly requested.

Do not add unnecessary GROUP BY columns.

============================================================
TEAM
============================================================

For a team being measured:

WHERE BattingTeam = 'team'

Do not treat a team as a batting team when the user says
"against", "vs", or "versus".

============================================================
PARTNERSHIP QUESTIONS
============================================================

If the user asks for the partnership between two players:

Example:

How many runs were scored in the partnership of Gambhir and Uthappa?

Use:

__METRIC_partnership_runs__

The partnership condition must match both batting orientations:

(
    batter = 'PLAYER_A'
    AND "non-striker" = 'PLAYER_B'
)

OR

(
    batter = 'PLAYER_B'
    AND "non-striker" = 'PLAYER_A'
)

The metric placeholder is already a complete aggregate expression.

WRONG:

SELECT SUM(__METRIC_partnership_runs__)

CORRECT:

SELECT __METRIC_partnership_runs__

Use total partnership runs, not only batsman's runs.

The application provides the actual metric expression.

============================================================
OPPONENT / AGAINST QUESTIONS
============================================================

When the user says:

"against Team"
"vs Team"
"versus Team"
"against the Team"

the mentioned team is the OPPONENT.

It is NOT the player's batting team.

Example:

runs scored by Gambhir against RCB

Correct interpretation:

- batter = Gambhir
- opponent = Royal Challengers Bangalore

Do NOT generate:

WHERE batter = 'G Gambhir'
AND BattingTeam = 'Royal Challengers Bangalore'

That means Gambhir batted for RCB, which is a different question.

This dataset has no direct opponent column.

To identify matches against an opponent, use the match ID.

Correct structure:

WHERE batter = 'G Gambhir'
AND ID IN (
    SELECT DISTINCT ID
    FROM ipl_ball_by_ball_cleaned
    WHERE BattingTeam = 'Royal Challengers Bangalore'
)

The same principle applies to:

runs against
fours against
sixes against
strike rate against
wickets against
economy against

The requested registered metric must still use its metric placeholder.

The opponent condition is applied through the shared match ID.

============================================================
PLAYER INNINGS
============================================================

For a player's innings breakdown:

GROUP BY ID, innings

For an innings filter:

WHERE innings = number

============================================================
SCHEMA RULES
============================================================

Use the exact supplied column names.

If a column contains a hyphen or space, use double quotes.

For this dataset:

"non-striker"

is correct.

non_striker

is incorrect.

Do not invent alternate column names.

============================================================
SAFETY
============================================================

Use only SELECT or WITH queries.

Never modify data.

Never read external files.

Return JSON only.

Ready response:

{
  "status": "ready",
  "sql": "...",
  "reason": null
}

Unsupported response:

{
  "status": "unsupported",
  "sql": null,
  "reason": "..."
}
"""


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):
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

def detect_requested_metrics(question):

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

        occupied.append(
            (start, end)
        )

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

def detect_entity_mentions(question):

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

    player_surnames = set()

    for player in players:

        normalized_player = normalize(player)

        parts = normalized_player.split()

        if len(parts) >= 2:

            surname = parts[-1]

            if len(surname) >= 4:
                player_surnames.add(surname)

    for index, word in enumerate(clean_words):

        normalized_word = normalize(word)

        if normalized_word not in player_surnames:
            continue

        candidate_results = []

        for size in range(
            min(3, index + 1),
            0,
            -1,
        ):

            start = index - size + 1

            candidate_words = clean_words[
                start:index + 1
            ]

            candidate = " ".join(
                candidate_words
            ).strip()

            if not candidate:
                continue

            result = resolve_entity(
                candidate,
                "player",
            )

            if result.get("status") == "resolved":

                candidate_results.append(
                    {
                        "value": candidate,
                        "score": result.get(
                            "score",
                            0,
                        ),
                        "length": size,
                    }
                )

        if candidate_results:

            best = max(
                candidate_results,
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

def resolve_question_entities(question):

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

    return (
        resolved_question,
        resolutions,
    )


# ============================================================
# SCHEMA CONTEXT
# ============================================================

def build_schema_context(schema):

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
    requested_metrics
):

    lines = [
        "AVAILABLE METRIC PLACEHOLDERS:"
    ]

    for metric in requested_metrics:

        if metric in METRIC_SQL_EXPRESSIONS:

            role = METRIC_ROLES.get(
                metric,
                "unknown",
            )

            lines.append(
                f"- {metric}: "
                f"__METRIC_{metric}__ "
                f"(role: {role})"
            )

    return "\n".join(lines)


# ============================================================
# METRIC PLACEHOLDER REPLACEMENT
# ============================================================

def replace_metric_placeholders(sql):

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
# NESTED METRIC AGGREGATE REPAIR
# ============================================================

def normalize_metric_placeholder_wrappers(sql):

    result = sql

    for metric_name in METRIC_SQL_EXPRESSIONS:

        placeholder = (
            f"__METRIC_{metric_name}__"
        )

        wrappers = [

            rf"SUM\s*\(\s*"
            rf"{re.escape(placeholder)}"
            rf"\s*\)",

            rf"AVG\s*\(\s*"
            rf"{re.escape(placeholder)}"
            rf"\s*\)",

            rf"COUNT\s*\(\s*"
            rf"{re.escape(placeholder)}"
            rf"\s*\)",

            rf"MIN\s*\(\s*"
            rf"{re.escape(placeholder)}"
            rf"\s*\)",

            rf"MAX\s*\(\s*"
            rf"{re.escape(placeholder)}"
            rf"\s*\)",
        ]

        for pattern in wrappers:

            result = re.sub(
                pattern,
                placeholder,
                result,
                flags=re.IGNORECASE,
            )

    return result


# ============================================================
# SCHEMA COLUMN NAME REPAIR
# ============================================================

def repair_schema_column_names(
    sql,
    schema,
):

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

        if re.search(
            r"[^A-Za-z0-9_]",
            actual_name,
        ):

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
# SQL VALIDATION
# ============================================================

def validate_sql(sql):

    if not isinstance(sql, str):

        raise ValueError(
            "Generated SQL must be a string."
        )

    cleaned = sql.strip()

    if not cleaned:

        raise ValueError(
            "Generated SQL is empty."
        )

    statements = [
        statement.strip()
        for statement in cleaned.split(";")
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
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_parquet",
        "httpfs",
        "parquet_scan",
        "glob",
    ]

    for function_name in external_functions:

        if (
            f"{function_name}("
            in normalized
        ):

            raise ValueError(
                "External file/network access "
                "is not allowed."
            )

    return statement


# ============================================================
# METRIC PLACEHOLDER VALIDATION
# ============================================================

def validate_metric_placeholders(
    sql,
    requested_metrics,
):

    missing = []

    validated_metrics = set(
        METRIC_SQL_EXPRESSIONS.keys()
    )

    for metric in requested_metrics:

        if metric not in METRIC_SQL_EXPRESSIONS:
            continue

        if metric not in validated_metrics:
            continue

        placeholder = (
            f"__METRIC_{metric}__"
        )

        if placeholder not in sql:

            missing.append(metric)

    return missing


# ============================================================
# LLM PLANNER
# ============================================================

def call_planner(
    resolved_question,
    schema_context,
    metric_context,
):

    prompt = f"""
{SYSTEM_PROMPT}

============================================================
DATASET SCHEMA
============================================================

{schema_context}

============================================================
METRIC CONTEXT
============================================================

{metric_context}

============================================================
USER QUESTION
============================================================

{resolved_question}

Generate the SQL structure.

For every registered metric that is requested,
use the exact metric placeholder.

Do not calculate the metric yourself.

Return JSON only.
"""

    content = call_gemini(
        SYSTEM_PROMPT,
        prompt,
        2500,
    )

    if not content:

        raise ValueError(
            "LLM returned an empty response."
        )

    try:

        result = json.loads(content)

    except json.JSONDecodeError as error:

        raise ValueError(
            "LLM returned invalid JSON."
        ) from error

    return result


# ============================================================
# MAIN SQL GENERATION PIPELINE
# ============================================================

def generate_sql(question):

    if not question.strip():

        return {
            "status": "unsupported",
            "sql": None,
            "resolved_question": question,
            "reason": "Question is empty.",
        }

    # --------------------------------------------------------
    # Resolve player/team names
    # --------------------------------------------------------

    (
        resolved_question,
        resolutions,
    ) = resolve_question_entities(
        question
    )

    # --------------------------------------------------------
    # Inspect actual dataset schema
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
    # Detect semantic metrics
    # --------------------------------------------------------

    requested_metrics = (
        detect_requested_metrics(
            resolved_question
        )
    )

    metric_context = (
        build_metric_context(
            requested_metrics
        )
    )

    # --------------------------------------------------------
    # Ask Gemini to construct SQL
    # --------------------------------------------------------

    result = call_planner(
        resolved_question,
        schema_context,
        metric_context,
    )

    status = result.get(
        "status",
        "unsupported",
    )

    if status != "ready":

        return {
            "status": "unsupported",
            "sql": None,
            "resolved_question": (
                resolved_question
            ),
            "reason": result.get(
                "reason",
                "Question is unsupported.",
            ),
        }

    sql = result.get(
        "sql"
    )

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

    sql = repair_schema_column_names(
        sql,
        schema,
    )

    # --------------------------------------------------------
    # Verify registered metric placeholders
    # --------------------------------------------------------

    missing_placeholders = (
        validate_metric_placeholders(
            sql,
            requested_metrics,
        )
    )

    # --------------------------------------------------------
    # Metric repair
    # --------------------------------------------------------

    if missing_placeholders:

        repair_prompt = f"""
The SQL generated for this question is
semantically incomplete.

============================================================
USER QUESTION
============================================================

{resolved_question}

============================================================
GENERATED SQL
============================================================

{sql}

============================================================
REQUESTED METRICS
============================================================

{", ".join(requested_metrics)}

============================================================
REQUIRED RULES
============================================================

You MUST replace metric arithmetic with exact placeholders:

__METRIC_<metric_name>__

Examples:

__METRIC_runs__
__METRIC_strike_rate__
__METRIC_wickets__
__METRIC_fours__
__METRIC_sixes__
__METRIC_partnership_runs__

Do not calculate these metrics yourself.

Do not wrap metric placeholders in:

SUM()
AVG()
COUNT()
MIN()
MAX()

For partnership questions, use:

__METRIC_partnership_runs__

A partnership between PLAYER_A and PLAYER_B must match both:

(
    batter = 'PLAYER_A'
    AND "non-striker" = 'PLAYER_B'
)

OR

(
    batter = 'PLAYER_B'
    AND "non-striker" = 'PLAYER_A'
)

For opponent / against questions:

"against Team"

means Team is the opponent.

Do NOT use:

BattingTeam = 'Team'

as the player's team.

Instead use match ID:

ID IN (
    SELECT DISTINCT ID
    FROM ipl_ball_by_ball_cleaned
    WHERE BattingTeam = 'Team'
)

Use the actual schema column:

"non-striker"

not:

non_striker

Keep the original filters, grouping,
ordering and limit.

Return JSON only:

{{
  "status": "ready",
  "sql": "...",
  "reason": null
}}
"""

        content = call_gemini(
            SYSTEM_PROMPT,
            repair_prompt,
            800,
        )

        if not content:

            raise ValueError(
                "Metric repair returned "
                "an empty response."
            )

        try:

            result = json.loads(content)

        except json.JSONDecodeError as error:

            raise ValueError(
                "Metric repair returned "
                "invalid JSON."
            ) from error

        sql = result.get(
            "sql"
        )

        if not sql:

            raise ValueError(
                "Metric repair did not "
                "return SQL."
            )

        sql = normalize_metric_placeholder_wrappers(
            sql
        )

        sql = repair_schema_column_names(
            sql,
            schema,
        )

    # --------------------------------------------------------
    # Validate generated SQL
    # --------------------------------------------------------

    sql = validate_sql(
        sql
    )

    # --------------------------------------------------------
    # Replace authoritative metric placeholders
    # --------------------------------------------------------

    sql = replace_metric_placeholders(
        sql
    )

    # --------------------------------------------------------
    # Validate final SQL again
    # --------------------------------------------------------

    sql = validate_sql(
        sql
    )

    return {
        "status": "ready",
        "sql": sql,
        "resolved_question": (
            resolved_question
        ),
        "reason": None,
    }


# ============================================================
# CLI TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "IPL Copilot - SQL Query Planner"
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