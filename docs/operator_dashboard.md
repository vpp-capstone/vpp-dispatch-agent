# 운영 대시보드와 자동 계산

```bash
python -m streamlit run dashboard/app.py
```

화면을 열면 30초마다 **공유 파일 내용·ESS·REC 설정·최적화 코드**의 변경을 확인한다. 바뀐 경우에만
비교 백테스트(`forecast_backtest`)와 48시간 운영계획(`operating_plan`)을 다시 풀고, 검산을
통과한 결과를 모든 화면의 공통 결과로 올린다. 화면을 닫으면 확인하지 않는다. 버튼 없이 한 번
실행하려면 다음 명령을 쓴다.

```bash
python -m vpp.operator_pipeline
```

설정은 `config/operator_pipeline.yaml` 한 곳에서 정한다. 모델 학습·추론, Git commit/push,
다운로드, 외부 전송은 하지 않는다. 실행 기록은 `data/processed/operator_pipeline/status.json`.

## 팀원 모델이 바뀌었을 때

모델 학습·추론은 각 담당자가 돌리고 **결과 파일만 팀 Drive 로 공유**한다. 받은 파일을 아래
위치에 같은 이름으로 두면 다음 확인 때 자동으로 다시 계산된다. 파일명이 같아도 내용이 바뀌면
새 실행으로 구분하고, 옛 결과는 보존한다.

| 공유 파일 | 두는 위치 | 담당 |
|---|---|---|
| `solar_*_predictions.parquet`, `wind_*_predictions.parquet` | `artifacts/predictions/` | 발전량 |
| `smp_forecast.csv` (`smp_metrics.csv`, 비교용 `_b1`/`_actual`/`_18` 변형) | `data/processed/forecast/` | SMP |

발전량 parquet 에 `improved_generation_kwh` 열이 있으면 그 열을, 없으면 `lightgbm_generation_kwh`
열을 쓴다 (`forecast_inputs.normalize_generation_model`). 어느 열을 썼는지는 결과 묶음의
`forecast_columns` 와 실행 상태 화면에 적힌다. SMP 파일의 모델 버전은 `trained_until` 열로 본다.

설정의 `solar`/`wind` 가 가리키는 파일명이 바뀌면(예: 2025년 `*_ensemble_test_predictions.parquet`)
`config/operator_pipeline.yaml` 의 경로를 그 파일명으로 바꾼다.

## 화면과 결과의 의미

- 운영 관제 / 운영 계획: 발전·판매·충방전·SOC를 시간별로 재생.
- 예측 비교: 전체·태양광·풍력·SMP의 예측과 실측을 같은 시각으로 재생.
- 수익 분석: 두 계획의 누적 금액, SMP·REC·열화비, ESS 없음 대비 추가수익.
- 실행 상태: 공유 파일 유무, 계산 단계, 오류, 입력·코드 지문.

모든 화면은 같은 검산 완료 묶음의 스냅샷을 사용한다. 운영계획의 시작 잔량은 비교 계산에 있는
계획 잔량이며 실제 운전 잔량이 아니다. 실측으로 재최적화한 금액은 실제 실행 수익이 아니다.
실행 운전·편차 정산, 실제 SOC 이월, REC 초기 저장분 제외와 종료 SOC 정책 변경은 포함되지 않는다.
`generation_forecast.infer_improved` 는 발전량 담당이 학습 완료 모델로 예측 parquet 을 뽑을 때 쓰는
도구이며 이 실행 경로에서는 호출하지 않는다.
