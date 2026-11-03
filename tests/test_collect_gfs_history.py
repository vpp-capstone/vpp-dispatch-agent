import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.weather.collect_gfs_history import (
    DEFAULT_GRIDS,
    FORECAST_HORIZON_HOURS,
    LAST_LEAD_HOUR,
    load_collection_grids,
    select_candidate_dates,
    variables_for_hour,
)


class GfsHistoryTests(unittest.TestCase):
    def test_jeju_collection_grid_has_six_points(self) -> None:
        grids = load_collection_grids(DEFAULT_GRIDS)
        self.assertEqual(len(grids), 6)
        self.assertIn(
            (33.25, 126.50),
            set(zip(grids["latitude"], grids["longitude"])),
        )

    def test_variable_schedule(self) -> None:
        self.assertEqual(variables_for_hour(20), ("shortwave_radiation_interval_average",))
        self.assertIn("temperature_2m", variables_for_hour(21))
        self.assertNotIn("temperature_2m", variables_for_hour(22))
        self.assertIn("wind_u_80m", variables_for_hour(44))
        self.assertIn("total_cloud_cover", variables_for_hour(45))
        self.assertIn("wind_u_80m", variables_for_hour(68))
        self.assertIn("shortwave_radiation_interval_average", variables_for_hour(68))
        self.assertNotIn("wind_u_80m", variables_for_hour(69))
        self.assertNotIn("shortwave_radiation_interval_average", variables_for_hour(69))
        self.assertIn("total_cloud_cover", variables_for_hour(69))
        self.assertEqual(LAST_LEAD_HOUR, 68)
        self.assertEqual(FORECAST_HORIZON_HOURS, 48)

    def test_normal_resume_skips_known_failures(self) -> None:
        candidates, skipped = select_candidate_dates(
            date(2023, 5, 23),
            date(2023, 5, 25),
            {date(2023, 5, 24)},
            retry_failed_only=False,
        )
        self.assertEqual(candidates, [date(2023, 5, 23), date(2023, 5, 25)])
        self.assertEqual(skipped, 1)

    def test_explicit_retry_selects_only_known_failures(self) -> None:
        candidates, skipped = select_candidate_dates(
            date(2023, 5, 23),
            date(2023, 5, 25),
            {date(2023, 5, 24)},
            retry_failed_only=True,
        )
        self.assertEqual(candidates, [date(2023, 5, 24)])
        self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
