"""담당: 데이터&수요예측팀(이재원) — 전력 시장·수요 데이터의 수집과 정리.

원본마다 모듈이 하나씩 있고, 마지막에 build 가 그 결과들을 합친다.
사용법과 출력 형식은 docs/data_contract.md 의 "2. 전력 시장 및 수요" 참고.

    demand         시간별 제주 전력수요 (포털 CSV 여러 개)
    day_ahead      하루전 SMP·수요예측 (OpenAPI)
    realtime_xlsx  실시간 SMP (KPX 연도별 엑셀)
    realtime_api   실시간 SMP·수요예측 (제주시범사업정보 OpenAPI)
    curtailment    제주 풍력·태양광 출력제어 실적 (포털 CSV)
    build          위 결과를 합쳐 15분 표와 시간별 표 생성

시각 컬럼 ts 는 모두 KST 기준 구간 시작 시각이다 (원본 1시 = 00:00~01:00 -> 00:00).
"""

from pathlib import Path

RAW_DIR = Path("data/raw/market")
PROCESSED_DIR = Path("data/processed/market")
