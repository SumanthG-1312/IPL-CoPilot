from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Resolve the dataset relative to the project root (the parent of
# src/), NOT the current working directory, so the backend works no
# matter where uvicorn / tests are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = Path(
    os.getenv(
        "IPL_DATASET_PATH",
        PROJECT_ROOT / "data" / "ipl_ball_by_ball_cleaned.csv",
    )
)

TABLE_NAME = "ipl_ball_by_ball_cleaned"


# ============================================================
# ERRORS
# ============================================================
# Both subclass the built-in types the old code raised, so existing
# `except ValueError` / `except RuntimeError` handlers keep working,
# while callers (e.g. main.py) can now tell the two cases apart.

class SQLValidationError(ValueError):
    """The SQL was rejected before reaching DuckDB."""


class SQLExecutionError(RuntimeError):
    """DuckDB accepted the SQL but failed while running it."""


# ============================================================
# CONNECTION
# ============================================================

@lru_cache(maxsize=1)
def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Create the DuckDB connection and load the dataset only once.

    After the table is built, the database is locked down at the
    ENGINE level:

      * enable_external_access = false
            -> no reading/writing files, no URLs, no extensions.
               This also blocks DuckDB "replacement scans" such as
               SELECT * FROM 'data/some_file.csv', which string
               filters cannot reliably catch.
      * lock_configuration = true
            -> generated SQL cannot flip those settings back.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("Loading IPL dataset into DuckDB...")

    # keep_default_na / na_filter are unchanged on purpose: the
    # verified metric values depend on 'None' staying a plain string.
    df = pd.read_csv(
        DATASET_PATH,
        keep_default_na=False,
        na_filter=False,
    )

    connection = duckdb.connect(database=":memory:")

    connection.register("_ipl_dataframe", df)

    connection.execute(
        f"""
        CREATE TABLE "{TABLE_NAME}" AS
        SELECT *
        FROM _ipl_dataframe
        """
    )

    connection.unregister("_ipl_dataframe")

    # Lock down AFTER loading (loading needs file/dataframe access).
    connection.execute("SET enable_external_access = false")
    connection.execute("SET lock_configuration = true")

    print(
        f"IPL dataset loaded: "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    return connection


def warm_up() -> None:
    """
    Load the dataset eagerly (call from FastAPI startup) so the
    first user request does not pay the CSV-loading cost.
    """
    get_connection()


# ============================================================
# SQL VALIDATION
# ============================================================
# Defence in depth: the engine lock above is the real wall; this
# validator gives fast, readable rejections and blocks things the
# engine would still allow (e.g. PRAGMA, multiple statements).

# Statement-level keywords that must never appear as real SQL tokens.
FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "merge", "copy", "export", "import", "attach",
    "detach", "pragma", "install", "load", "call", "set",
    "reset", "vacuum", "checkpoint", "use", "execute", "prepare",
    "deallocate", "begin", "commit", "rollback", "grant", "revoke",
)

_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b"
)

# Table functions / helpers that touch files, URLs or the host.
_EXTERNAL_FUNCTION_RE = re.compile(
    r"\b("
    r"read_\w+|\w+_scan|glob|httpfs|sniff_csv|"
    r"getenv|current_setting|duckdb_\w+|pragma_\w+"
    r")\s*\("
)

# FROM/JOIN directly followed by a quoted string = file path.
_FILE_SOURCE_RE = re.compile(r"\b(from|join)\s*'")

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_QUOTED_IDENT_RE = re.compile(r'"(?:[^"]|"")*"')


def _mask_sql(sql: str) -> str:
    """
    Return a lower-cased copy of the SQL with comments removed and
    string literals / quoted identifiers blanked out, so keyword
    checks only look at real SQL tokens. This prevents false alarms
    like a player named 'Update Singh' or the "non-striker" column,
    and prevents ';' inside a literal from splitting statements.
    """

    text = _COMMENT_RE.sub(" ", sql)

    # Mark file-path style sources BEFORE literals get blanked.
    # (Detected separately in validate_sql on the comment-free text.)
    text = _STRING_LITERAL_RE.sub("''", text)
    text = _QUOTED_IDENT_RE.sub('"x"', text)

    return " ".join(text.lower().split())


def validate_sql(sql: str) -> str:
    """
    Allow only a single, read-only SELECT / WITH statement.

    Returns the cleaned statement (without trailing semicolon).
    """

    if not isinstance(sql, str):
        raise SQLValidationError("SQL must be a string.")

    cleaned = sql.strip()

    if not cleaned:
        raise SQLValidationError("SQL query is empty.")

    # Strip comments once; keep literals for the real statement.
    no_comments = _COMMENT_RE.sub(" ", cleaned).strip()

    masked = _mask_sql(cleaned)

    # ---- single statement (semicolons inside literals ignored) ----
    statements = [
        part for part in masked.split(";") if part.strip()
    ]

    if len(statements) != 1:
        raise SQLValidationError(
            "Only one SQL statement is allowed."
        )

    # ---- must start with SELECT or WITH ----
    if not re.match(r"^\s*(select|with)\b", masked):
        raise SQLValidationError(
            "Only SELECT or WITH queries are allowed."
        )

    # ---- forbidden keywords (real tokens only) ----
    match = _FORBIDDEN_RE.search(masked)

    if match:
        raise SQLValidationError(
            f"Forbidden SQL operation: {match.group(1)}"
        )

    # ---- external file / host access ----
    if _EXTERNAL_FUNCTION_RE.search(masked):
        raise SQLValidationError(
            "External file access is not allowed "
            "inside generated SQL."
        )

    if _FILE_SOURCE_RE.search(
        _COMMENT_RE.sub(" ", cleaned).lower()
    ):
        raise SQLValidationError(
            "Querying files directly is not allowed. "
            f"Use the {TABLE_NAME} table."
        )

    # Return the statement without a trailing semicolon, keeping
    # original literals/identifiers intact.
    return no_comments.rstrip().rstrip(";").strip()


# ============================================================
# EXECUTION
# ============================================================

def execute_sql(sql: str) -> pd.DataFrame:
    """
    Execute read-only SQL against the persistent in-memory table.

    Each call uses its own cursor, so concurrent FastAPI worker
    threads never share one connection object (a DuckDB connection
    is not safe for simultaneous use from several threads).
    """

    validated_sql = validate_sql(sql)

    cursor = get_connection().cursor()

    try:
        return cursor.execute(validated_sql).fetchdf()

    except Exception as error:
        raise SQLExecutionError(
            f"DuckDB query execution failed: {error}"
        ) from error

    finally:
        cursor.close()


def close_connection() -> None:
    """
    Close the cached DuckDB connection.

    Mainly useful for testing or application shutdown.
    """

    if get_connection.cache_info().currsize:
        get_connection().close()

    get_connection.cache_clear()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    print("IPL Copilot - SQL Executor")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter SQL: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            break

        if not query:
            continue

        try:
            result = execute_sql(query)

            print("\nResult:")
            print(result.to_string(index=False))

        except Exception as error:
            print("\nExecution Error:")
            print(error)