
from pathlib import Path
import sys
from statistics import median
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_executor import (
    DATASET_PATH,
    TABLE_NAME,
    execute_sql,
    get_connection,
    validate_sql,
)

REPEATS = 15 
WARMUPS = 2

QUERIES = {
    "row_count": f"""
        SELECT COUNT(*) AS row_count
        FROM "{TABLE_NAME}"
    """,
    "total_runs": f"""
        SELECT SUM(total_run) AS total_runs
        FROM "{TABLE_NAME}"
    """,
    "top_10_batters": f"""
        SELECT batter, SUM(batsman_run) AS runs
        FROM "{TABLE_NAME}"
        GROUP BY batter
        ORDER BY runs DESC, batter ASC
        LIMIT 10
    """,
}


def load_reference():
    df = pd.read_csv(
        DATASET_PATH,
        keep_default_na=False,
        na_filter=False,
    )

    for column in ("total_run", "batsman_run"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def expected_results(df):
    top_batters = (
        df.groupby("batter", as_index=False)["batsman_run"]
        .sum()
        .rename(columns={"batsman_run": "runs"})
        .sort_values(
            ["runs", "batter"],
            ascending=[False, True],
        )
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "row_count": pd.DataFrame(
            {"row_count": [len(df)]}
        ),
        "total_runs": pd.DataFrame(
            {"total_runs": [df["total_run"].sum()]}
        ),
        "top_10_batters": top_batters,
    }


def results_match(actual, expected):
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)

    if list(actual.columns) != list(expected.columns):
        return False

    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
        return True
    except AssertionError:
        return False


def percentile(values, percentile_value):
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def main():
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("IPL COPILOT - SQL BENCHMARK")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Repetitions per query: {REPEATS}")
    print("\nLoading independent Pandas reference...")

    reference_start = perf_counter()
    df = load_reference()
    reference_load_seconds = perf_counter() - reference_start
    expected = expected_results(df)

    print(f"Reference rows: {len(df):,}")
    print(
        f"Reference CSV load: "
        f"{reference_load_seconds:.3f} seconds"
    )

    print("\nMeasuring DuckDB initialization...")
    cold_start = perf_counter()
    get_connection()
    cold_start_seconds = perf_counter() - cold_start

    print(
        f"DuckDB initialization: "
        f"{cold_start_seconds:.3f} seconds"
    )

    rows = []

    for name, sql in QUERIES.items():
        print(f"\nBenchmarking: {name}")

        # Validate correctness before collecting timings.
        result = execute_sql(sql)
        correct = results_match(result, expected[name])

        # Warm-up runs are excluded from latency statistics.
        for _ in range(WARMUPS):
            execute_sql(sql)

        timings = []

        for _ in range(REPEATS):
            start = perf_counter()
            execute_sql(sql)
            timings.append((perf_counter() - start) * 1000)

        row = {
            "query": name,
            "correct": correct,
            "median_ms": round(median(timings), 3),
            "p95_ms": round(percentile(timings, 0.95), 3),
            "min_ms": round(min(timings), 3),
            "max_ms": round(max(timings), 3),
            "repeats": REPEATS,
        }
        rows.append(row)

        print(f"  Correct: {correct}")
        print(f"  Median:  {row['median_ms']:.3f} ms")
        print(f"  P95:     {row['p95_ms']:.3f} ms")
        print(f"  Min:     {row['min_ms']:.3f} ms")
        print(f"  Max:     {row['max_ms']:.3f} ms")

    # Measure SQL validation separately.
    validation_sql = QUERIES["total_runs"]
    validation_timings = []

    for _ in range(100):
        start = perf_counter()
        validate_sql(validation_sql)
        validation_timings.append(
            (perf_counter() - start) * 1000
        )

    report = pd.DataFrame(rows)
    report.to_csv("benchmark_results.csv", index=False)

    accuracy = (
        sum(row["correct"] for row in rows)
        / len(rows)
        * 100
    )

    print("\n" + "=" * 48)
    print("BENCHMARK SUMMARY")
    print("=" * 48)
    print(f"SQL test cases: {len(rows)}")
    print(f"Correct test cases: {sum(r['correct'] for r in rows)}")
    print(f"Test-case accuracy: {accuracy:.2f}%")
    print(f"DuckDB initialization: {cold_start_seconds:.3f} s")
    print(
        "Median SQL validation: "
        f"{median(validation_timings):.4f} ms"
    )
    print("\nResults saved to benchmark_results.csv")
    print(report.to_string(index=False))

    print(
        "\nNote: This measures the SQL execution path, "
        "not full LLM-to-answer latency."
    )


if __name__ == "__main__":
    main()
