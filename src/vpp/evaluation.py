"""예측 평가: 예측·대시보드팀 / 운영비용 평가: 최적화·대시보드팀. 모든 계산은 담당팀이 구현한다."""

import pandas as pd

from vpp.schemas import CostEvaluation, ForecastEvaluation, ForecastResponse, OptimizationResult


def evaluate_forecast(forecast: ForecastResponse, actuals: pd.DataFrame) -> ForecastEvaluation:
    """TODO: 지표·산식·결측 처리 합의 후 실측 비교와 5% 목표 판정을 구현한다."""
    raise NotImplementedError("미구현: 예측·대시보드팀이 예측 평가를 구현해 주세요.")


def evaluate_cost(plan: OptimizationResult, actuals: pd.DataFrame) -> CostEvaluation:
    """TODO: 고정된 계획을 실측에 적용한 비용과 동일 조건 기준 방식의 비용을 비교한다."""
    raise NotImplementedError("미구현: 최적화·대시보드팀이 운영비용 평가를 구현해 주세요.")
