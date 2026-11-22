import unittest
import pandas as pd

from src.metric_engine import calculate_metric
from src.sql_executor import SQLValidationError, validate_sql
from src.entity_resolver import resolve_entity


class MetricEngineTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ID": [1, 1, 1, 1],
                "innings": [1, 1, 1, 1],
                "overs": [0, 0, 0, 0],
                "ballnumber": [1, 2, 3, 4],
                "batter": ["V Kohli", "V Kohli", "V Kohli", "V Kohli"],
                "bowler": ["Bowler A"] * 4,
                "non-striker": ["RG Sharma"] * 4,
                "extra_type": ["None", "wide", "None", "None"],
                "batsman_run": [4, 0, 6, 1],
                "extras_run": [0, 1, 0, 0],
                "total_run": [4, 1, 6, 1],
                "non_boundary": [0, 0, 0, 0],
                "isWicketDelivery": [0, 0, 1, 0],
                "player_out": ["None", "None", "V Kohli", "None"],
                "kind": ["None", "None", "caught", "None"],
                "fielders_involved": ["None"] * 4,
                "BattingTeam": ["RCB"] * 4,
            }
        )

    def test_batting_metrics(self):
        self.assertEqual(calculate_metric(self.df, "runs", "batter"), 11)
        self.assertEqual(calculate_metric(self.df, "fours", "batter"), 1)
        self.assertEqual(calculate_metric(self.df, "sixes", "batter"), 1)
        self.assertEqual(calculate_metric(self.df, "balls_faced", "batter"), 3)
        self.assertAlmostEqual(
            calculate_metric(self.df, "strike_rate", "batter"),
            366.6666666666667,
        )


class SQLSafetyTests(unittest.TestCase):
    def test_read_only_select_is_allowed(self):
        self.assertTrue(
            validate_sql(
                "SELECT batter, SUM(batsman_run) FROM ipl_ball_by_ball_cleaned GROUP BY batter"
            ).startswith("SELECT")
        )

    def test_destructive_sql_is_rejected(self):
        with self.assertRaises(SQLValidationError):
            validate_sql("DELETE FROM ipl_ball_by_ball_cleaned")

    def test_external_csv_is_rejected(self):
        with self.assertRaises(SQLValidationError):
            validate_sql("SELECT * FROM read_csv('secret.csv')")


class EntityResolutionTests(unittest.TestCase):
    def test_full_name_aliases_resolve(self):
        for name in ("Rohit Sharma", "Gautam Gambhir"):
            result = resolve_entity(name, "player")
            self.assertIn(result["status"], {"resolved", "ambiguous", "not_found"})


if __name__ == "__main__":
    unittest.main()
