"""담당: 데이터&수요예측팀 — 데이터 수집·전처리·로딩을 직접 구현할 공간.

입출력 초안은 docs/data_contract.md 참고. 현재 파일 읽기나 데이터 변환도 수행하지 않는다.
"""

from pathlib import Path

import pandas as pd


def prepare_hourly(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    """TODO: 수집 방법·시간 정렬·단위 변환·결측 정책을 정하고 정제 데이터를 저장한다.

    입력: 원본 폴더, 저장 폴더. 출력: 팀이 합의한 시간별 DataFrame.
    수집 함수와 특징 생성 함수는 담당팀이 필요에 따라 추가한다.
    """
    raise NotImplementedError("미구현: 데이터팀이 수집·전처리를 구현해 주세요.")


def load_hourly(path: Path) -> pd.DataFrame:
    """TODO: 팀이 정한 CSV/Parquet 형식의 정제 데이터 로딩과 검증을 구현한다."""
    raise NotImplementedError("미구현: 데이터팀이 데이터 로딩을 구현해 주세요.")
