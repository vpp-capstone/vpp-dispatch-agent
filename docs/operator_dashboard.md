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

### 데이터 묶음 여러 개 (예: 2025 예측 + 2024 실측)

`config/operator_pipeline.yaml` 의 `datasets:` 에 묶음을 여러 개 적으면 묶음마다 MILP·실행 결과를
따로 계산해 두고, 화면 사이드바의 **데이터 묶음** 선택으로 바꿔 본다. 첫 묶음이 기본이다. 파일이 없는
묶음은 그 묶음의 단계만 "파일 대기"가 되고, 다른 묶음과 그 묶음의 이전 결과는 그대로 남는다. 팀원이 두
파일을 다 가지고 있으면 2025 예측 계획과 2024 실측 기반 실행 결과를 같은 화면에서 번갈아 볼 수 있다.

발전량 담당의 기본 구성은 세 묶음이다.

- `2025 최종 평가`: 동결된 운영 예측값에 2025 실측만 결합한다. 모델을 다시 선택하거나 튜닝하지 않는다.
- `2025 운영 입력`: 실측 열이 없는 원래 운영용 예측으로, 계획 생성만 검증한다.
- `2024 개선 검증`: 선택된 개선 구조를 2023년 라벨만으로 다시 학습해 2024년을 예측한다. 개발
  검증 자료이므로 독립 최종 테스트로 해석하지 않는다.

`python -m vpp.generation_forecast.export_evaluation_predictions`가 2024 검증 파일은
`artifacts/2024_validation/`, 2025 평가 파일은 `artifacts/2025/evaluation/`에 만들고 SHA-256
요약은 `artifacts/generation_evaluation_export_summary.json`에 기록한다. 실측 없는 2025 운영
파일은 `artifacts/2025/operational/`에 둔다. parquet은 Git에 넣지 않고 Drive로 공유한다.

## 화면과 결과의 의미

- 요약: 오늘 판매계획, 어제 실적(실행 수익 또는 실측 대기), 이번 달 누적 예상 수익, ESS 효과.
- 운영계획: 하루 KPI, 발전·판매·충방전·SOC 재생, 시간별 판매계획 표·CSV. 실측이 있으면 아래에
  **계획 대비 실행**(확정 판매 vs 실제 공급, 부족·초과, 계획·실행 충방전, 계획·실행 SOC, 전날 실행
  종료 SOC → 오늘 시작)이 붙는다. 시뮬레이터의 재생 선택에 "확정 계획을 실측에 적용한 실행 결과"가 생긴다.
- 예측 입력: 그날 예측 발전량·SMP. 실측이 있으면 같은 시각 비교 재생과 오차.
- 예상 수익: 집계 기간별 금액 분해. 실측이 있으면 계산 기준 세 가지를 따로 고른다 —
  (1) 예측으로 세운 계획의 예상 수익, (2) 확정 계획을 실측에 적용한 실행 수익, (3) 실측을 미리 알고
  세운 비교 계획. 셋을 합치거나 서로 대신 쓰지 않는다.
- 실적·정산: 정산 항목별 계획값·실행값·상태(가정 기반 실행 수익 / 실험 가정 / 미구현 / 미연결)와
  기간 전체의 세 운영 방식 비교(ESS 미사용 · 기존 계획 방식 · 수정 계획 + 실행 운전).
- 실행 상태: 공유 파일 유무, 계산 단계, 오류, 입력·코드 지문.

모든 화면은 같은 검산 완료 묶음의 스냅샷을 사용한다. 실측이 있는 기간에는 화면의 '계획'이 **전날 실행
종료 잔량에서 시작한 확정 계획**이고, 48시간 운영계획의 시작 잔량도 그 실행 종료 잔량이다
(`start_soc_source=executed`). 실측이 없는 기간(현재 2025)은 계획상 잔량에서 시작하며 실행·정산은
"미계산"으로 표시한다. 실행 수익은 `docs/execution_settlement.md` 의 실험용 정산 가정(하루전·편차 가격
= 실측 SMP, 배수 1·벌칙 0)으로 계산한 **가정 기반 실행 수익**이며 실제 정산이 아니다. REC 초기 저장분
제외, 발전량 예측 정산금, 용량정산금은 포함되지 않는다.
`generation_forecast.infer_improved` 는 발전량 담당이 학습 완료 모델로 예측 parquet 을 뽑을 때 쓰는
도구이며 이 실행 경로에서는 호출하지 않는다.
