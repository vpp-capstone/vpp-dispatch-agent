import unittest
from datetime import date

import pandas as pd

from vpp.generation_forecast.dataset import solar_elevation_deg
from vpp.generation_forecast.weather_quality import EXPECTED_ROWS, validate_daily_frame


class GenerationTrainingMartTests(unittest.TestCase):
    def test_solar_elevation_day_is_above_night(self) -> None:
        times = pd.Series(pd.to_datetime(["2024-06-21 03:00:00Z", "2024-06-21 15:00:00Z"]))
        latitude = pd.Series([33.5, 33.5])
        longitude = pd.Series([126.5, 126.5])
        elevation = solar_elevation_deg(times, latitude, longitude)
        self.assertGreater(elevation[0], 60)
        self.assertLess(elevation[1], 0)

    def test_daily_validation_accepts_complete_frame(self) -> None:
        planning = date(2024, 1, 1)
        valid = pd.date_range("2024-01-01", periods=48, freq="h", tz="Asia/Seoul")
        frame = pd.DataFrame(
            [
                {
                    "gfs_grid_id": f"g{grid}",
                    "issue_time_utc": pd.Timestamp("2023-12-30 18:00:00Z"),
                    "issue_time_kst": pd.Timestamp("2023-12-31 03:00:00", tz="Asia/Seoul"),
                    "bid_deadline_kst": pd.Timestamp("2023-12-31 11:00:00", tz="Asia/Seoul"),
                    "available_before_bid_deadline": True,
                    "valid_time_utc": timestamp.tz_convert("UTC"),
                    "valid_time_kst": timestamp,
                    "lead_hour": 21 + offset,
                    "shortwave_radiation_w_m2": 0.0,
                    "temperature_2m_c": 10.0,
                    "total_cloud_cover_pct": 50.0,
                    "wind_u_80m_ms": 1.0,
                    "wind_v_80m_ms": 2.0,
                }
                for grid in range(6)
                for offset, timestamp in enumerate(valid)
            ]
        )
        result = validate_daily_frame(frame, planning)
        self.assertEqual(len(frame), EXPECTED_ROWS)
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
