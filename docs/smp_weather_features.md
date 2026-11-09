# SMP 예측용 제주 기상 피처

## 목적

발전량 예측에 사용한 6개 제주 GFS 격자의 48시간 예보를 제주 하루전 SMP 모델에 전달할
시간별 피처로 집계한다. 원본 GFS와 발전량 학습마트는 수정하지 않으며, 별도의 CSV를 만든다.

## 생성

저장소 루트에서 실행한다.

```bash
python -m vpp.smp_weather_features
```

기본 범위는 2023-01-01~2025-12-31이고 결과는 다음 위치에 생성된다.

```text
data/processed/weather/
  jeju_renewable_weather_features.csv
  jeju_renewable_weather_features_summary.json
```

날짜나 경로를 바꿀 때는 다음 옵션을 사용한다.

```bash
python -m vpp.smp_weather_features \
  --start 2023-01-01 --end 2025-12-31 \
  --input data/raw/gfs_history_core/daily \
  --plants data/reference/plant_location_master_21.csv \
  --output data/processed/weather/jeju_renewable_weather_features.csv
```

`--strict`를 붙이면 요청 기간에 누락되거나 품질검사를 통과하지 못한 날짜가 하나라도 있을 때
실패한다. 현재 원본에는 2023-05-24가 없으므로 전체 기간 실행에는 `--strict`를 사용하지 않는다.

## 시간 계약

| 열 | 의미 |
|---|---|
| `planning_day_kst` | 48시간 예측의 첫 대상일 |
| `issued_at` | GFS 기준시각, KST |
| `available_at` | 6개 격자 파일이 모두 준비된 가장 늦은 시각, KST |
| `target_ts` | 기상 예측 대상 시간, KST |

SMP 모델은 `target_ts`가 같은 후보 중 `available_at <= D-1 11:00`인 가장 최근 예보만 사용한다.
집계값 하나를 만들려면 6개 격자가 모두 필요하므로 `available_at`은 같은 대상 시간에 대한
`source_file_last_modified_utc`의 최댓값으로 계산한다. 이렇게 해야 일부 격자가 아직 도착하지
않은 시각으로 잘못 기록되지 않는다.

## 피처

제주 6격자 전체에 대해 다음 통계를 만든다.

- 일사량: 평균·최대·표준편차
- 2m 기온: 평균·최소·최대·표준편차
- 전운량: 평균·최소·최대·표준편차
- 80m 풍속: 평균·최소·최대·표준편차
- 80m U·V 성분 평균과 평균 벡터 풍속

또한 `plant_location_master_21.csv`의 설비용량을 격자별로 합산해 태양광 18개소와 풍력
3개 단지 위치 가중 피처를 각각 만든다.

- 일사량·기온·전운량·80m U·V·풍속의 설비용량 가중 평균
- 풍력 발전 민감도를 표현하기 위한 80m 풍속 세제곱의 설비용량 가중 평균

이 가중치는 제주 전체 설비 구성이 아니라 현재 프로젝트의 가상 포트폴리오 21개소 기준이다.
따라서 제주 전체 SMP를 설명하는 지역 평균 피처와 함께 사용하고, 포트폴리오 가중 피처만으로
제주 전체 재생에너지 출력을 대표한다고 해석하지 않는다.

`weather_grid_coverage`, `weather_grid_coverage_ratio`, `quality_*` 열은 전달 품질 확인용이다.
이름에 `coverage` 또는 `quality`가 포함되므로 `vpp.smp_forecast`가 모델 입력에서 자동 제외한다.

## 현재 결과

2023~2025 실행 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 요청 계획일 | 1,096일 |
| 처리 계획일 | 1,095일 |
| 누락 | 2023-05-24 |
| 품질 실패일 | 0일 |
| 출력 행 | 52,560행 |
| 모델 입력 피처 | 32개 |
| `available_at + target_ts` 중복 | 0건 |
| 숫자 피처 결측 | 0건 |
| D-1 11시 이후 가용 처리된 행 | 0건 |

## SMP 모델에서 사용

시장 정리 파일이 준비된 환경에서 기상 미사용 결과와 사용 결과를 각각 실행해 비교한다.

```bash
python -m vpp.smp_forecast
python -m vpp.smp_forecast \
  --weather data/processed/weather/jeju_renewable_weather_features.csv
```

필수 시장 입력은 `data/processed/market/market_hourly.csv`와
`data/processed/market/smp_forecast_hourly.csv`다.
