# 제주 VPP 기상예보 기반 발전량 예측 — GitHub 이전 인수인계서

> 작성 기준: 2026-11-03 KST  
> 기존 로컬 작업 폴더: `capstone_data`  
> 목적: 팀 GitHub 레포와 연결된 새 Codex가 현재 상태를 오해하지 않고 개발을 이어가기 위한 문서

## 1. 가장 먼저 확인할 사항

팀 레포 README의 현재 설명은 **제주 전력수요를 다음 날 24시간 예측**하는 구조다. 반면 이 로컬 폴더에서 구현한 것은 **기상예보를 이용해 태양광·풍력 발전량을 48시간 예측**하는 모듈이다.

따라서 기존 `predict_demand`를 발전량 예측으로 덮어쓰면 안 된다. 첫 PR 전에 다음 사항을 팀과 합의하고 `docs/data_contract.md` 및 `src/vpp/schemas.py`에 반영해야 한다.

- 수요예측과 발전량예측을 서로 다른 모듈·스키마로 유지할지
- 발전량 예측 결과를 MILP와 SMP 모델에 어떤 단위로 전달할지
- 기존 README의 24시간 설명을 유지할지, 발전량 모듈만 48시간으로 둘지
- 모델 입력 시점은 `D-1 11:00 KST` 입찰 마감 이전으로 고정한다는 규칙

권장 구조는 `src/vpp/generation_forecast/`에 별도 패키지를 만들고, `src/vpp/forecast.py`에는 공통 파이프라인이 호출할 얇은 연결 함수만 두는 것이다.

## 2. 현재 모듈의 목적과 범위

### 예측 목적

- 제주 VPP 중개사업자의 입찰·ESS 운영계획에 사용할 발전량 예측
- 계획 시점: 거래일 전날 오전 11시 KST 이전
- 예측 범위: 계획일 00:00부터 다음 날 23:00까지 총 48시간
- 시간 해상도: 1시간
- 발전원: 태양광과 풍력을 별도 모델로 예측
- 현재 정답 범위: 제주 전체가 아니라 확보된 태양광 18개소와 풍력 3개 단지

### SMP 팀과의 관계

현재 발전량 정답은 제주 전체 발전량이 아니므로 이 모델의 예측 합계를 **제주 전체 재생에너지 발전량으로 간주하면 안 된다**. SMP 모델에는 같은 제주 6개 GFS 격자의 기상값을 제주 전역 가상 피처로 전달할 수 있지만, 21개 발전소의 발전량 합계는 표본 포트폴리오 예측이다.

KPX 제주 집계 발전량은 본 발전량 모델의 주 정답에서 제외했다. 특히 풍력은 2024년 중반 이후 시장·입찰제도 및 공개 범위 변화로 시계열 수준이 급감하고, 태양광·풍력 거래량에는 ESS 충·방전이 포함될 수 있어 순수 발전량 정답으로 바로 쓰기 어렵다.

## 3. 데이터 구성

### GFS 과거 예보

- 소스: NOAA GFS 과거 예보
- 공간: 제주를 대표하는 0.25도 격자 6개
- 예보 계획일별 행 수: `6개 격자 × 48시간 = 288행`
- 표준 사용 사이클: `D-2 18 UTC`
- 표준 lead: 21~68시간
- 목표 시간: D 00:00 ~ D+1 23:00 KST
- 주요 변수:
  - 시간별 하향 단파복사량
  - 80m U/V 바람 성분 및 파생 풍속·풍향
  - 2m 기온
  - 전운량
- GFS의 3시간 간격 기온·운량은 시간 단위로 선형 보간한다.
- 각 자료는 입찰 마감 전에 실제 공개되었는지 확인하는 규칙을 둔다.

### 발전량 정답

- 태양광: 한수원 제주 자료의 18개 발전소, 시간별 발전량
- 풍력: 한경·성산·상명 3개 단지, 시간별 발전량
- 위치·설비용량·GFS 격자 연결: `plant_location_master_21.csv`
- 태양광과 풍력은 서로 다른 LightGBM 모델을 사용한다.
- 학습 목표는 `capacity_factor = generation_kwh / capacity_kw`이며 예측 후 설비용량을 곱해 kWh로 복원한다.

### 풍력 정답 정제 규칙

- 한경·성산은 단계별 값 1·2를 합산한다.
- 한경의 중복 충돌 행은 임의 선택하지 않고 `label_is_valid=False`로 둔다.
- 상명은 2025-02-01 이전 원자료를 Wh로 보고 `/1000`하여 kWh로 통일하고, 이후 값은 kWh로 처리한다.
- 결측·단계 충돌·설비용량 초과 값은 삭제하지 않고 품질 플래그와 함께 보존하되 모델 학습에서는 제외한다.

## 4. 현재 구현된 파이프라인

실행 순서는 다음과 같다.

1. `collect_gfs_history.py`: 계획일별 GFS 6격자·48시간 자료 수집
2. `validate_gfs_history.py`: 날짜별 288행, 격자·시간·중복·결측·입찰마감 준수 검사
3. `build_generation_training_mart.py`: 발전량 정제 후 GFS 예보와 결합
4. `train_generation_models.py`: 기준 모델과 태양광·풍력 LightGBM 학습 및 평가

`collect_gfs_history.py`는 이미 저장된 날짜를 건너뛰므로 중단 후 같은 명령으로 재개할 수 있다. `failed_dates.json`에 기록된 날짜는 일반 재개에서 반복 시도하지 않으며, 명시적인 실패 재시도 옵션을 쓸 때만 다시 처리한다.

## 5. 학습·검증 규칙과 데이터 누수 방지

- 분할 기준은 **계획일이 아니라 실제 예측 대상 시각인 `valid_time_kst`의 연도**다.
- 2023년: 학습
- 2024년: 검증·조기 종료·모델 선택
- 2025년: 최종 테스트 전용
- 동일한 실제 발전 시각이 연말 경계에서 학습과 검증 양쪽에 들어가지 않도록 한다.
- 발전량 지연값은 D-1 11시 시점에 알 수 없는 미래 발전량이 섞일 위험 때문에 1차 모델에서 사용하지 않았다.
- 모델 학습에는 반드시 `label_is_valid == True`인 행만 사용한다.
- 고유키는 `asset_id + issue_time_utc + valid_time_utc`다. 동일 실제 시각이 서로 다른 계획일 예보창에 포함될 수 있으므로 `asset_id + valid_time`만으로 중복 제거하면 안 된다.
- 태양광은 야간 또는 단파복사량 0일 때 예측값을 0으로 제한한다.

## 6. 현재까지의 결과

### 2023~2024 GFS 품질검사

- 예상 계획일: 731일
- 정상 계획일: 730일
- 누락: `2023-05-24` 한 날짜
- 커버리지: 99.863201%
- 정상 파일의 행 수·격자·시간·중복·결측·마감 후 공개 오류: 0건

`2023-05-24`의 기본 18 UTC GFS 파일은 D-1 11:00 KST 이후 공개되어 의도적으로 실패한다. 같은 날짜를 계속 재시도하지 않도록 처리했지만, **D-2 12 UTC 사이클을 사용하는 fallback 로직은 아직 구현하지 않았다**. fallback을 구현할 경우 한 계획일 안에서 서로 다른 GFS 사이클을 섞으면 안 되며, 사용 사이클과 fallback 사유를 결과에 남겨야 한다.

### 학습마트

- 태양광: 630,720행, 18개소, 유효 정답 100%
- 풍력: 105,120행, 3개 단지, 유효 정답 99.901065%
- 풍력 품질 이슈는 결측, 단계 충돌, 설비용량 초과이며 원본 값을 임의 보정하지 않았다.

### 2024 검증 성능

아래 수치는 모델 선택에 사용한 **검증 결과**이며 최종 테스트 성능이 아니다.

| 발전원 | 모델 | 포트폴리오 48시간 nMAE | 24시간차 | 48시간차 |
|---|---:|---:|---:|---:|
| 태양광 | 기준 모델 | 6.2063% | 6.2061% | 6.2064% |
| 태양광 | LightGBM | 3.7941% | 3.6679% | 3.9202% |
| 풍력 | 기준 모델 | 13.8128% | 13.8128% | 13.8128% |
| 풍력 | LightGBM | 6.4347% | 6.1199% | 6.7495% |

- 태양광 기준 대비 개선율: 38.87%
- 풍력 기준 대비 개선율: 53.42%
- 태양광 최적 반복: 175
- 풍력 최적 반복: 106

## 7. 2025 수집 진행 상태

2026-11-03 15:12 KST 스냅샷 기준:

- GFS 일별 파일: 883개
- 범위: 2023-01-01 ~ 2025-06-02
- 전체 목표: 2025-12-31까지
- 수집은 문서 작성 시점에도 진행 중일 수 있으므로 이전 직전에 실제 최대 날짜를 다시 확인한다.

확인 명령:

```powershell
$files = Get-ChildItem .\weather_pipeline\gfs_history_core\daily -Filter 'gfs_*.parquet' -File -Recurse
$files.Count
$files.BaseName | Sort-Object | Select-Object -Last 1
```

재개 명령:

```powershell
.\.venv\Scripts\python.exe -m scripts.weather.collect_gfs_history `
  --start 2023-01-01 `
  --end 2025-12-31
```

## 8. 다음 개발 순서

1. 현재 GFS 수집을 2025-12-31까지 완료한다.
2. 2023~2025 전체 기간에 대해 GFS 품질검사를 다시 실행한다.
3. 2023~2025 학습마트를 다시 생성한다.
4. 저장된 현재 모델로 2025년을 먼저 평가한다. 이때 재학습·튜닝·특징 변경을 하지 않는다.
5. 2025 최종 테스트 지표와 예측 파일을 확정한다.
6. 그 후에만 2023~2024 전체로 운영 모델을 재학습한다.
7. MILP 전달용으로 다음 컬럼의 예측 인터페이스를 만든다.

```text
asset_id
energy_type
issue_time_utc
issue_time_kst
bid_deadline_kst
valid_time_utc
valid_time_kst
lead_hour
forecast_day_offset
predicted_generation_kwh
model_version
```

발전소별 결과와 별도로 `energy_type + issue_time + valid_time` 기준 포트폴리오 합계를 제공하는 것이 좋다.

## 9. GitHub 레포로 옮길 파일

### 반드시 커밋할 코드

아래 로컬 파일을 기능에 맞게 옮긴다. 팀 레포 구조를 지키기 위해 루트에 그대로 쌓기보다는 권장 경로로 이동한다.

| 로컬 파일 | 팀 레포 권장 위치 | 비고 |
|---|---|---|
| `collect_gfs_dayahead_sample.py` | `scripts/weather/collect_gfs_dayahead_sample.py` | GFS 요청·파싱 공통 함수 포함 |
| `collect_gfs_history.py` | `scripts/weather/collect_gfs_history.py` | 전체 과거 예보 수집기 |
| `validate_gfs_history.py` | `src/vpp/generation_forecast/weather_quality.py` | import 경로 수정 필요 |
| `build_generation_training_mart.py` | `src/vpp/generation_forecast/dataset.py` | 기본 데이터 경로를 팀 구조에 맞게 수정 |
| `train_generation_models.py` | `src/vpp/generation_forecast/model.py` | 학습·평가 로직 |
| `tests/test_collect_gfs_dayahead_sample.py` | 동일한 `tests/` | import 경로 수정 |
| `tests/test_collect_gfs_history.py` | 동일한 `tests/` | import 경로 수정 |
| `tests/test_generation_training_mart.py` | 동일한 `tests/` | import 경로 수정 |
| `tests/test_train_generation_models.py` | 동일한 `tests/` | import 경로 수정 |

주의: 현재 스크립트의 기본 경로는 대부분 `Path(__file__).resolve().parent`를 기준으로 한다. 파일 위치만 바꾸면 기본 경로가 깨지므로 팀 레포의 `data/raw`, `data/processed`, `artifacts` 구조에 맞게 경로 설정을 먼저 분리해야 한다.

### 커밋할 소형 기준정보

| 로컬 파일 | 팀 레포 권장 위치 |
|---|---|
| `jeju_gfs_grids_6.csv` | `data/reference/jeju_gfs_grids_6.csv` |
| `plant_location_master_21.csv` | `data/reference/plant_location_master_21.csv` |
| `태양광18개_풍력3개_데이터확인결과.md` | `docs/generation_asset_validation.md` |
| `발전량 예측 AI 모델 설계.md` | `docs/generation_forecast_design.md` |
| 이 인수인계서 | `docs/generation_forecast_handoff.md` |

`plant_location_master_21.csv`에는 공개 가능한 위치·설비 정보만 있는지 마지막으로 확인한 후 커밋한다. 민감한 실제 VPP 고객 자산정보로 교체될 경우 Git에서 제외해야 한다.

### 커밋 가능한 재현용 샘플·요약

전체 데이터 대신 다음 소형 파일만 커밋하는 것을 권장한다.

- `weather_pipeline/gfs_sample/gfs_day_ahead_20250103.parquet`
- `weather_pipeline/gfs_sample/gfs_day_ahead_20250103_validation.json`
- `weather_pipeline/gfs_sample/gfs_day_ahead_20250103_validation.md`
- `weather_pipeline/gfs_sample/plant_to_gfs_grid_mapping.parquet`
- `weather_pipeline/model_results/generation_model_metrics.csv`
- `weather_pipeline/model_results/generation_asset_metrics.csv`
- `weather_pipeline/model_results/solar_feature_importance.csv`
- `weather_pipeline/model_results/wind_feature_importance.csv`
- `weather_pipeline/model_results/training_summary.json`

권장 위치는 각각 `data/sample/generation_forecast/`와 `docs/results/`다. 단, 팀 정책상 Parquet 샘플도 금지한다면 CSV 1일 샘플로 변환한다.

### `pyproject.toml`에 합칠 의존성

현재 기능에 직접 필요한 핵심 의존성은 다음과 같다.

- `numpy`
- `pandas`
- `pyarrow`
- `requests`
- `scikit-learn`
- `joblib`
- `lightgbm>=4.7`
- 개발 의존성: `pytest`

현재 로컬 `requirements.txt`에는 과거 KMA·ECMWF 실험용 `xarray`, `netCDF4`, `pyproj`, `cfgrib`, `eccodes`, `python-dotenv`도 들어 있다. 현 GFS 발전량 파이프라인에서 실제로 쓰는지 확인하지 않고 모두 팀 의존성에 추가하지 않는다.

## 10. Git에 넣지 않고 팀 Drive로 공유할 파일

팀 README 규칙에 따라 아래는 Git 커밋 금지다.

- `발전량 데이터/`의 원본 CSV 4개
- `weather_pipeline/gfs_history_core/` 전체 과거 GFS Parquet과 로그
- `weather_pipeline/model_data/*.parquet` 전체 학습마트
- `weather_pipeline/model_results/*.joblib` 학습 모델
- `weather_pipeline/model_results/*predictions.parquet` 전체 검증 예측
- `.env`

Drive 공유 시 다음 구조를 권장한다.

```text
generation_forecast/
  raw/generation/
  raw/gfs_history_core/
  processed/model_data/
  artifacts/models/
  manifests/
```

각 공유 묶음에는 파일 목록, 기간, 크기, SHA-256 체크섬을 함께 둔다. Git 문서에는 Drive 경로와 버전명만 기록한다.

## 11. 옮기지 않아도 되는 파일

현재 본선 GFS 발전량 예측과 직접 관계없는 과거 실험·중간 파일은 팀 레포로 복사하지 않는다.

- `.venv/`, `__pycache__/`
- `.env`
- `archive/obsolete_kma_pipeline_20261030/`
- `collect_ecmwf_ifs_history.py`와 `weather_pipeline/ecmwf_ifs_history/`
- `solar_irradiance_model.py`
- `renewables_hourly.csv`
- `CLEANUP_20261030.md`, `ECMWF_IFS_COLLECTION.md`
- 과거 행정구역 매핑 파일 `jeju_forecast_regions.csv`, `jeju_forecast_region_grid_mapping.csv`
- 보고서 제출물 `TR-3조-김리나-20221531.docx`
- 원본 제주 발전소 현황 CSV와 위치 교차검증 중간 파일

이 자료가 연구 근거로 필요하면 Git이 아니라 Drive의 `archive/`에 보관하고, 현재 운영 코드와 섞지 않는다.

## 12. `.gitignore`에 추가할 항목

팀 레포의 기존 규칙을 유지하면서 최소한 다음을 확인한다.

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
data/raw/
data/processed/
artifacts/models/
artifacts/predictions/
*.joblib
weather_pipeline/
```

샘플 데이터와 요약 지표를 커밋할 경우 상위 폴더 전체를 무시하지 말고 `!data/sample/**` 같은 예외 규칙을 팀 구조에 맞게 둔다.

## 13. 새 Codex에게 전달할 작업 지시

새 Codex는 다음 순서로 작업해야 한다.

1. 팀 README, `docs/data_contract.md`, `src/vpp/schemas.py`, `src/vpp/forecast.py`를 먼저 읽는다.
2. 수요예측 24시간과 발전량예측 48시간의 개념 충돌을 확인하고 기존 함수를 덮어쓰지 않는다.
3. 이 문서의 코드만 별도 `generation_forecast` 모듈로 이식한다.
4. 데이터 경로를 코드에서 분리하고 CLI 또는 설정 객체로 받도록 한다.
5. 소형 샘플로 테스트가 통과한 뒤에만 Drive 원자료를 연결한다.
6. 2025년을 손대지 않은 최종 테스트로 먼저 평가한다.
7. 최종평가 후 MILP 전달 스키마를 구현한다.
8. API 키, 원본 데이터, 모델 파일, 전체 예측 결과는 커밋하지 않는다.

## 14. 알려진 주의사항

- 2024 검증 점수를 최종 성능이라고 쓰면 안 된다.
- 2025 데이터를 보고 하이퍼파라미터나 특징을 바꾸면 최종 테스트의 의미가 사라진다.
- 발전소 21개의 합계를 제주 전체 발전량이라고 표현하면 안 된다.
- GFS는 관측값이 아니라 당시 이용 가능했던 예보값이어야 한다.
- 입찰 마감 이후 공개된 예보를 학습에 사용하면 시간 누수다.
- 풍향은 각도 평균보다 U/V 성분을 사용해야 한다.
- 48시간차 성능은 24시간차보다 나쁘므로 MILP가 이를 동일한 확실성으로 취급하지 않도록 horizon 정보를 넘긴다.
- 현재 가상환경에는 `pytest`가 설치되어 있지 않아 2026-11-03 현장 재실행은 `No module named pytest`로 실패했다. 팀 레포 개발 의존성에 `pytest`를 포함하고 PR 전에 전체 테스트를 다시 실행해야 한다.

