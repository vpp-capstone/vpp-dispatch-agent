# 발전량·가격 예측 → ESS 입력 검증 (박시영)

## 맡는 작업과 경계

현재 채택된 태양광 18곳·풍력 3곳의 실측/예측 자료가 ESS 최적화 입력으로 전달될 때
발전소 범위, 시간, 단위, 누락 처리 기준이 맞는지 확인한다.
발전량 모델·SMP 모델·ESS 목적함수·대시보드 연결은 기존 담당자의 구현을 사용한다.
이 문서는 별도 검증 도구의 실행 기준이며 `schemas.py`나 팀 공통 데이터 계약을 변경하지 않는다.

1. 원본 CSV 4개의 파일명·크기·SHA-256·기간을 기록한다.
2. 기존 `build_portfolio_hourly`를 호출해 시간별 실측 표와 제외 날짜 목록을 생성한다.
3. 예측 학습용 발전량 로더와 ESS 로더가 같은 시간의 동일한 발전량을 만드는지 비교한다.
4. 발전량/가격 예측 파일의 누락·중복·단위·48시간 연속성을 검사한다.
5. 실제 예측 파일이 확보되면 연결 입력을 생성하고 통합 담당자에게 검증 결과를 전달한다.
6. 실제 수익 평가는 통합 담당자의 실행 방식과 실행 시점 보정 정책을 확인한 후 진행한다.

## 기존 구현에서 확인한 입력 기준

| 자료 | 현재 파일/열 | 시간/단위 및 주의 |
|---|---|---|
| 태양광 실측 | `제주#1_기상정보 융합 태양광 발전 시계열 데이터.csv` | 발전기 18곳, 계량전력량 kWh. 원본 1시를 KST 00:00 시작 구간으로 변환 |
| 성산·한경 실측 | 남부발전 원본 CSV 2개 | 단계 1+2를 합산, kWh. 결측 단계는 0으로 대체하지 않음 |
| 상명 실측 | 중부발전 상명풍력 원본 CSV | 기존 두 로더 모두 2025-01-31까지 Wh÷1,000→kWh, 이후 kWh 적용. 원자료 단위 변화에 대한 기존 가정 |
| 발전량 예측 | `solar_validation_predictions.parquet`, `wind_validation_predictions.parquet` | `asset_id`, `issue_time_utc`, `valid_time_utc`, `lightgbm_generation_kwh`. `generation_kwh`는 사후 실측 정답이므로 계획 입력에서 제외 |
| SMP 예측 | `smp_forecast.csv` | `issue_time`, `target_time`, `smp_hat`(원/kWh), `smp_hat_krw_per_mwh`(원/MWh). `smp_actual`은 계획 입력에서 제외 |
| 발전소 기준 | `data/reference/plant_location_master_21.csv` | 정확한 발전소 ID 집합으로 검사. 제주 전체 발전량과 구분 |
| ESS 입력 | 생성하는 `forecast_windows.csv` | KST 시작시각 `ts`, `generation_mwh`, `price_krw_per_mwh`, `dt_hours=1`. 계획일별 48행 |

- 서로 다른 예측 모듈의 발행시각은 같지 않을 수 있다. **KST 계획일 + 대상 구간 시각**으로 결합한다.
- 현재 GFS 수집 코드의 기준 예보시각은 계획 대상일 전날 03:00 KST,
  SMP 발행시각은 전날 11:00 KST다. 각 모듈의 `lead_hour`가 달라도 정상일 수 있다.
- 검사는 발행시각 기준 마감 후 자료를 차단하지만 **실제 자료 공개 지연/전처리 완료시각까지 증명하지 않는다.**
- 한 계획일의 입력은 다음 날 00:00부터 그다음 날 23:00까지 정확히 48시간이다.
  기본값 `--incomplete-windows reject`는 시간/발전소 누락도 오류로 보고한다.
  검증용 자료에서 불완전한 창 전체를 제외하려면 `--incomplete-windows exclude`를 명시한다.
  제외 날짜·행 수·사유를 남기며, 중복·잘못된 시각/단위·비정상 발전량은 이 옵션으로 숨기지 않는다.
- 소스별 유효한 계획일의 차집합은 보고서에 기록한다. 공통 계획일이 하나도 없으면 실패다.
- 음수·0원 SMP는 삭제하지 않는다. kWh→MWh는 ÷1,000, 원/kWh→원/MWh는 ×1,000이다.
- 학습용 로더의 발전소별 품질 제외와 ESS 로더의 날짜 제외/플래그 정책은 별개다.
  이번 검증은 값을 비교하며 두 정책을 임의로 통일하지 않는다.

## 실행

Python 3.11 이상 환경에서 프로젝트 설치 후 저장소 루트에서 실행한다.
원본은 기존 팀 공유 폴더에서 받아 `data/raw/portfolio/`에 파일명 그대로 넣는다.

```bash
pip install -e ".[dev]"
python -m vpp.handoff_audit --start 2024-06-01 --end 2025-12-31
```

예측 파일까지 받았다면:

```bash
python -m vpp.handoff_audit --start 2024-06-01 --end 2025-12-31 --solar artifacts/predictions/solar_validation_predictions.parquet --wind artifacts/predictions/wind_validation_predictions.parquet --smp data/processed/forecast/smp_forecast.csv --incomplete-windows exclude
```

출력은 `data/processed/handoff_audit/`에 저장한다.

| 파일 | 용도 |
|---|---|
| `audit.json` | 입력 체크섬, 요청 기간, 제외 사유, 두 발전량 로더 값 비교, 예측 검증 상태 |
| `portfolio_actual_hourly.csv` | 기존 로더 정책으로 정리한 실측 표. 관측값을 예측값이라고 부르지 않음 |
| `excluded_days.csv` | 실측 입력에서 제외한 날짜와 사유 |
| `forecast_windows.csv` | 예측 3개를 모두 받았을 때만 생성. 계획일별로 선택해서 최적화 담당자에게 전달 |
| `incomplete_forecast_windows.csv` | 명시적으로 제외한 불완전 계획일·소스·행 수·사유. 제외가 없으면 헤더만 저장 |

종료코드: 0=입력 형식 연결 검사 완료, 1=검사 실패, 2=실측만 검사했고 예측 파일 대기.
종료코드 0도 실제 수익 평가·자료 공개시각·ESS 실시간 실행 가능성을 인증하지 않는다.
이전에 생성한 출력과 섞이지 않도록 실행별 `--out` 폴더를 권장한다.
같은 폴더로 재실행하면 이전 `audit.json`과 예측 출력 2개를 먼저 제거하여
실패한 실행 뒤에 과거 성공 예측 파일이 남지 않도록 한다.

## 공유 방법

- **GitHub:** `feat/data-handoff-audit` 브랜치에서 코드·테스트·소형 결과 요약만 공유한다.
  초안 PR에서 입력 처리 기준을 검토한 후 병합하며, 원본·전체 예측 결과는 포함하지 않는다.
- **팀 Drive/Teams/노션 첨부:** 실측 CSV, 예측 Parquet, 전체 생성 CSV/JSON은 데이터 묶음으로 공유한다.
  Git에는 원본·전체 데이터·모델·`.env`를 올리지 않는다.
- **스크린샷:** 이번 작업은 화면 변경이 없어 필수 아님. 검증 요약과 제외 날짜 CSV가 증거다.
- **필요한 인계:** 리나님 예측 Parquet 2개와 재원님 SMP CSV의 팀 공유 위치/버전을 확인한다.
  통합 담당자의 동일 파일 사본을 사용하고 새 데이터나 새 모델을 임의로 섞지 않는다.
- **발표 표현:** 이번 결과는 데이터 일치성 검증이다. 예측 정확도 개선이나 수익 개선으로 표현하지 않는다.

## 후속 작업 완료 조건

- [x] 실제 예측 파일 3개에 대해 검사 실행 및 체크섬 기록
- [x] 완전한 공통 계획 188개와 그 대상시각의 사후 실측 존재 확인 (공식 명령으로 재현)
- [ ] 규리님 통합 PR의 입력 값과 이번 검증 출력 값 대조
- [ ] 예측 오차 시 충전/방전 부족분 처리와 평가 조건 합의 후 수익 검증

예측 파일 확보 전의 실측 검증은 [1차 결과 요약](results/data_handoff_audit_20261106.md)에 보존한다.
실제 예측 파일 확보 이후의 결과는 [공유 드라이브 예측 검증](results/forecast_handoff_20261106.md)에 있다.
