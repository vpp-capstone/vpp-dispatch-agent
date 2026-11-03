import unittest
from datetime import date

from scripts.weather.collect_gfs_dayahead_sample import issue_for_valid_day, radiation_interval


class GfsDayAheadSampleTests(unittest.TestCase):
    def test_issue_cycle_covers_korean_day_with_leads_21_to_44(self) -> None:
        issue = issue_for_valid_day(date(2025, 1, 3))
        self.assertEqual(issue.isoformat(), "2025-01-01T18:00:00+00:00")

    def test_radiation_interval(self) -> None:
        description = "653:1:d=2025010118:DSWRF:surface:18-21 hour ave fcst:"
        self.assertEqual(radiation_interval(description), (18, 21))


if __name__ == "__main__":
    unittest.main()
