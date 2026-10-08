"""담당: 대시보드팀 — 담당팀이 완성한 모듈을 연결할 공간. 현재 실행 흐름도 미구현."""

import argparse
from datetime import date

from vpp.schemas import DayAheadRun, ForecastRequest


def run_day_ahead(request: ForecastRequest) -> DayAheadRun:
    """TODO: 데이터팀과 발행시점·입력을 합의하고 예측 → ESS → 평가를 연결한다.

    각 단계의 입출력·오류 처리·실측 도착 시점은 담당팀과 정한다.
    현재 데이터 로딩·선택·예측 호출·최적화 호출·평가 호출을 수행하지 않는다.
    """
    raise NotImplementedError("미구현: 대시보드팀이 담당팀의 완성 모듈을 연결해 주세요.")


def main() -> None:
    """실행 진입점만 유지. 미구현은 종료 코드 2로 알리고 결과를 만들지 않는다."""
    parser = argparse.ArgumentParser(description="제주 VPP 개발 뼈대 — 기능 미구현")
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    try:
        result = run_day_ahead(ForecastRequest(target_date=args.date))
    except NotImplementedError as exc:
        parser.exit(2, f"{exc}\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
