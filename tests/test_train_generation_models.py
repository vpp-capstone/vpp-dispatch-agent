import unittest

import numpy as np

from vpp.generation_forecast.model import error_metrics


class TrainGenerationModelsTests(unittest.TestCase):
    def test_error_metrics(self) -> None:
        result = error_metrics(
            np.array([10.0, 20.0]),
            np.array([12.0, 16.0]),
            np.array([100.0, 100.0]),
        )
        self.assertEqual(result["observations"], 2)
        self.assertAlmostEqual(result["mae_kwh"], 3.0)
        self.assertAlmostEqual(result["rmse_kwh"], np.sqrt(10.0))
        self.assertAlmostEqual(result["nmae_pct"], 3.0)
        self.assertAlmostEqual(result["mae_capacity_factor"], 0.03)


if __name__ == "__main__":
    unittest.main()
