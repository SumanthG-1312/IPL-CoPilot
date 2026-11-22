import unittest


class ExecutorCompatibilityTests(unittest.TestCase):
    def test_package_module_has_no_broken_import_contract(self):
        from src.sql_executor import validate_sql

        self.assertEqual(
            validate_sql("SELECT COUNT(*) FROM ipl_ball_by_ball_cleaned"),
            "SELECT COUNT(*) FROM ipl_ball_by_ball_cleaned",
        )

        with self.assertRaises(ValueError):
            validate_sql("DROP TABLE ipl_ball_by_ball_cleaned")


if __name__ == "__main__":
    unittest.main()
