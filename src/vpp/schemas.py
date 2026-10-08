"""전원 협의: 모듈 간 입출력 형식 초안. 기능 구현이나 데이터 처리 로직은 포함하지 않는다.

사용자가 정한 KST·1시간·24개 예측 요구를 기준으로 형태만 둔다.
시간 연속성·발행시점·데이터 품질 검증은 담당팀이 구현한다.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastRequest(Contract):
    target_date: date
    # TODO(전원): 입력 데이터/모델 식별자·발행시점 등 필요한 필드 합의


class ForecastPoint(Contract):
    timestamp: datetime  # KST 시간별 시각. 시간대·연속성 검증은 담당팀 구현
    demand_mw: float = Field(ge=0)


class ForecastResponse(Contract):
    target_date: date
    model: str
    is_sample: bool
    points: list[ForecastPoint] = Field(min_length=24, max_length=24)


class OptimizationRequest(Contract):
    forecast: ForecastResponse
    # TODO(최적화·대시보드팀): 발전량·가격·ESS 사양 등 필요한 입력 추가


class OptimizationResult(Contract):
    status: str
    reason: str = ""
    # TODO(최적화·대시보드팀): 계획·SOC 등 출력과 성공/실패 상태 정의


class ForecastEvaluation(Contract):
    status: str
    reason: str = ""
    # TODO(예측·대시보드팀): 지표·산식·오차·5% 목표 판정 필드 정의


class CostEvaluation(Contract):
    status: str
    reason: str = ""
    # TODO(최적화·대시보드팀): 실측 기반 비용·비교 기준·차이 필드 정의


class DayAheadRun(Contract):
    forecast: ForecastResponse
    optimization: OptimizationResult
    forecast_evaluation: ForecastEvaluation
    cost_evaluation: CostEvaluation
