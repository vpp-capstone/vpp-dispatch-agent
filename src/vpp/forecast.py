"""담당: 데이터&수요예측팀 — 예측·학습 로직을 직접 구현할 공간.

특정 기준선이나 모델은 구현하지 않았다. 일반 모델 구현 후 비용 인지 학습으로 확장한다.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from vpp.schemas import ForecastResponse


def predict_demand(
    inputs: pd.DataFrame, target_date: date, *, is_sample: bool
) -> ForecastResponse:
    """TODO: 사용 가능한 입력으로 대상일 KST 00~23시의 수요 24개를 예측한다.

    입력 특징·발행시점·모델 로딩 방식은 담당팀이 합의한다. 미래 실측은 입력에서 제외한다.
    출력: ForecastResponse. 현재 예측값을 생성하지 않는다.
    """
    raise NotImplementedError("미구현: 예측팀이 수요예측을 구현해 주세요.")


def train_baseline(features_path: Path, model_dir: Path) -> Path:
    """TODO: 일반 예측 모델 선택, 시간순 분할·학습·저장. 출력은 저장한 모델 경로."""
    raise NotImplementedError("미구현: 예측팀이 일반 모델 학습을 구현해 주세요.")


def train_cost_aware(features_path: Path, model_dir: Path) -> Path:
    """TODO: 예측·최적화팀 공동으로 비용 정의와 학습 연결 방식을 합의한 뒤 구현한다."""
    raise NotImplementedError("미구현: 예측·최적화팀이 비용 인지 학습을 구현해 주세요.")
