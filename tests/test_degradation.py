import numpy as np

from vpp.degradation import FixedThroughputModel, NoDegradationModel, build_degradation_model
from vpp.optimization import EssParams


def test_fixed_throughput_model_matches_plan_cost():
    model = FixedThroughputModel(100.0)
    charge, discharge = model.milp_coefficients(2)
    assert np.allclose(charge, [100.0, 100.0])
    assert np.allclose(discharge, [100.0, 100.0])
    assert model.cost_for_arrays([1.0, 2.0], [3.0, 4.0]) == 1000.0


def test_none_model_has_zero_cost():
    model = build_degradation_model("none", 0.0)
    assert isinstance(model, NoDegradationModel)
    assert model.cost_for_arrays([10.0], [20.0]) == 0.0


def test_v0_and_v01_configs_load_as_different_models():
    v0 = EssParams.from_yaml("config/ess_v0.yaml")
    v01 = EssParams.from_yaml("config/ess_v0_1.yaml")
    assert v0.degradation_model == "none"
    assert v01.degradation_model == "fixed_throughput"
    assert v0.throughput_cost_krw_per_mwh == 0.0
    assert v01.throughput_cost_krw_per_mwh > 0.0
