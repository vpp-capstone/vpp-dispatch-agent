"""담당: 최적화·대시보드팀 — ESS 최적화 로직을 직접 구현할 공간.

먼저 필요한 입출력 필드를 schemas.py에 정의한다. 사양·비용·제약·솔버는 미정이다.
"""

from vpp.schemas import OptimizationRequest, OptimizationResult


def plan_ess(request: OptimizationRequest) -> OptimizationResult:
    """TODO: 입력·ESS 사양·제약·비교 기준을 합의하고 운영계획과 솔버 상태를 반환한다.

    출력 필드(시간별 충·방전, SOC 등)도 담당팀이 정의한다. 현재 계산/결과 생성 없음.
    """
    raise NotImplementedError("미구현: 최적화팀이 ESS 운영계획을 구현해 주세요.")
