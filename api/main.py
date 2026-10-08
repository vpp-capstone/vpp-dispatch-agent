"""FastAPI (담당: 대시보드). 실행: python -m uvicorn api.main:app --reload → http://localhost:8000/docs

라우터는 입력 검증과 오류 응답만 맡고 계산은 src/vpp/에서 한다.
"""

from fastapi import FastAPI, HTTPException

from vpp.pipeline import run_day_ahead
from vpp.schemas import DayAheadRun, ForecastRequest

app = FastAPI(title="Jeju VPP Day-Ahead API", version="0.3.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/runs/day-ahead", response_model=DayAheadRun)
def day_ahead(request: ForecastRequest) -> DayAheadRun:
    """연결용 진입점. 현재는 501이며 담당자가 통합 실행 구현 후 결과를 반환한다."""
    try:
        return run_day_ahead(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
