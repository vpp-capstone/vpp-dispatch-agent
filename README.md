# 제주 VPP 발전량·SMP 예측 기반 ESS 운영계획

제주 VPP 중개사업자 입장에서, 입찰 마감인 **D-1 11시**에 알 수 있는 기상예보·시장자료만으로
**다음 48시간의 태양광·풍력 발전량과 제주 하루전 SMP를 1시간 단위로 예측**하고,
그 예측으로 ESS 충·방전 계획을 세운 뒤 실측으로 결과를 평가한다.

- 발전량: 한수원 태양광 18개소 + 풍력 3개 단지(성산·한경·상명)의 가상 포트폴리오. 제주 전체가 아니다.
- 현재 단계: 발전량·SMP 예측을 ESS v0에 연결해 **예측 기반 계획과 실측 기반(완전예측) 계획을 비교**하는 중.
  실측에 따른 실행 조정·정산은 다음 단계.
- 초기 기획의 **24시간 수요예측**(`data.py`, `forecast.py`)은 자리만 있고 구현되지 않았다. 유지 여부는 팀 합의 필요.

## 모듈 현황

| 모듈                   | 파일                                                                                   | 상태                                                                          | 문서                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 시장·수요 수집         | `vpp.market`                                                                           | 구현 (PR #1)                                                                  | [data_contract.md](docs/data_contract.md)                                                         |
| 재생에너지 Target 정제 | `renewable_targets.py`, `build_renewable_history.py`                                   | 구현 (PR #2)                                                                  | [renewable_history.md](docs/renewable_history.md)                                                 |
| ESS 운영계획 v0        | `optimization.py`, `backtest.py`, `portfolio_data.py`, `validation.py`, `baselines.py` | 구현 (PR #3). 실측을 넣는 완전예측 백테스트. 예측 연결 진행 중                | [formulation.md](docs/formulation.md)                                                             |
| 발전량 예측 48h        | `generation_forecast/`                                                                 | 구현. 2023 학습·2024 결합 학습·2025 최종 테스트, LightGBM 및 앙상블 지원      | [설계](<docs/generation_forecast_design(11-03).md>) · [결과](docs/generation_forecast_results.md) |
| SMP 예측 48h           | `smp_forecast.py`                                                                      | 구현 (PR #5)                                                                  | [smp_forecast.md](docs/smp_forecast.md)                                                           |
| 수요예측 24h           | `data.py`, `forecast.py`                                                               | 미구현 (`NotImplementedError` 자리만)                                         | —                                                                                                 |
| 평가·정산              | `evaluation.py` — `evaluate_cost`, `evaluate_forecast`                                 | 미구현. 예측 기반 계획을 실측에 적용한 손익 (다음 단계)                       | —                                                                                                 |
| 파이프라인·API·화면    | `pipeline.py`, `api/main.py`, `dashboard/app.py`                                       | 뼈대. 대시보드의 ESS 화면만 동작                                              | —                                                                                                 |
| 공통 타입              | `schemas.py`                                                                           | 초안. 최적화·평가 타입은 `status`, `reason`만 있는 빈 틀                      | —                                                                                                 |

- 공통 타입·함수 인자·데이터 형식은 바꾸기 전에 쓰는 사람과 합의.
- 파일이 커지면 패키지로 쪼개도 됨. import하는 쪽도 같이 고칠 것.

## 흐름

```text
vpp.market ──► data/processed/market/ ──► vpp.smp_forecast ──► SMP 예측 48h ─┐
                                                                             ├─► ESS 계획 (solve_window) ─► 평가·정산 (미구현)
data/raw/generation/ ─► vpp.generation_forecast ─► 발전량 예측 48h ───────────┘
         └──────────────── 실측 발전량·SMP (비교·평가용) ─────────────────────┘
```

1. 시장 수집 → SMP 예측, 발전량 원본 + GFS → 발전량 예측 (각자 독립 실행)
2. 같은 계획일 D의 두 예측 48시간을 ESS에 넣어 앞 24시간 계획 확정, SOC 이월
3. 같은 기간 실측 기반 계획과 비교 (현재 단계)
4. 확정 계획을 실측에 적용한 정산·손익 평가 (다음 단계)

## 데이터 준비 (설치 전에)

Git에 들어 있는 데이터는 `data/reference/`와 `data/sample/`뿐이다. 나머지는 팀 Drive에서 받아
아래 위치에 **파일명 그대로** 둔다. 전부 `.gitignore` 대상이라 커밋되지 않는다.

**1. 발전량 원본 CSV 4개 → `data/raw/generation/`** (ESS 백테스트·발전량 예측 공용, 필수)

```text
제주#1_기상정보 융합 태양광 발전 시계열 데이터.csv
한국남부발전(주)_성산풍력발전실적_20260331.csv
한국남부발전(주)_한경풍력발전실적_20260331.csv
한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv
```

- ESS 백테스트(`vpp.backtest`)와 발전량 예측 모두 이 폴더가 기본값이다.
- 발전량 예측은 파일명에 `태양광`·`성산풍력`·`한경풍력`·`상명풍력`이 하나씩 들어 있으면 찾는다.
  macOS에서 받은 한글 파일명은 자모가 분리(NFD)되어 못 찾을 수 있다. 그때는 파일명을 다시 입력해 저장한다.

**2. 예측 결과 (ESS에 예측을 연결할 때)**

| 받는 곳     | 파일                                                                          | 넣는 위치                             |
| ----------- | ----------------------------------------------------------------------------- | ------------------------------------- |
| 발전량 담당 | `*_validation_predictions.parquet`, `*_test_predictions.parquet`, `*_ensemble_*_predictions.parquet` | `artifacts/predictions/`              |
| 발전량 담당 | `*_lightgbm.joblib`, `*_ensemble.joblib`                                                         | `artifacts/models/`                   |
| 발전량 담당 | `generation_model_metrics.csv`, `generation_ensemble_metrics.csv` 등 지표·요약                    | `data/processed/generation_forecast/` |
| SMP 담당    | `smp_forecast.csv` (`smp_metrics.csv`)                                        | `data/processed/forecast/`            |

**3. 해당 모듈을 직접 돌릴 때만**

- 발전량 예측 학습: 과거 GFS → `data/raw/gfs_history_core/daily/year=YYYY/month=MM/gfs_YYYYMMDD.parquet`.
  Drive에서 받거나 `python -m scripts.weather.collect_gfs_history`로 수집.
- SMP 예측: 시장 정리본 `market_hourly.csv`, `smp_forecast_hourly.csv` → `data/processed/market/`.
- 시장 수집: 원본은 `data/raw/market/` 아래. 하위 폴더·출처는 [data_contract.md](docs/data_contract.md) 2번 표.
  OpenAPI는 `.env`의 `DATA_GO_KR_API_KEY` 필요.

## 설치·실행

운영자 화면과 로컬 모델 자동 실행은 [운영 대시보드 안내](docs/operator_dashboard.md)를 참고하세요.
`dashboard/app.py`는 코드·입력 변경을 확인해 예측 → MILP → 검산 → 화면 갱신을 연결합니다.
원본이 없으면 실행 상태에 누락 자료를 표시하며, 기존 예측 결과와 새 모델 결과를 구분합니다.

Python 3.11 이상.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

```bash
python -m streamlit run dashboard/app.py       # 화면 http://localhost:8501 (샘플 그래프)
python -m uvicorn api.main:app --reload        # API 문서 http://localhost:8000/docs
python -m vpp.pipeline --date 2026-01-03       # CLI (모듈 연결 후 동작)
pytest                                       # PR 전에 한 번
```

ESS 계산 CLI 3개(`vpp.operating_plan` 운영계획, `vpp.forecast_backtest` 예측 vs 완전예측 비교,
`vpp.backtest` 실측 백테스트)는 공통 옵션 `--ess config/ess_v0.yaml`,
`--degradation-config config/ess_v0_1.yaml`(열화비), `--rec config/rec_v0.yaml`(REC 수익 가정,
없으면 미적용)을 받는다. 대시보드 사이드바의 ESS·열화비·REC 선택이 같은 인수로 들어간다.
REC 는 판매 1 MWh 당 1회·단일 가중치·시나리오 단가의 **가정값**이며 ESS 없음 기준에도 같이 적용된다
(`docs/formulation.md` §2).

끄는 건 `Ctrl + C`.
`data/sample/hourly.csv`는 화면 확인용 샘플.
쓸 라이브러리(LightGBM, PuLP 등)는 정한 뒤 `pyproject.toml`의 `dependencies`에 추가.

## 폴더

제주 재생에너지 데이터는 공개 파일 수집 → 과거 파일 정제 → 팀 전달 CSV 생성 순서로 실행한다.
[재생에너지 팀 전달 문서](docs/renewable_history.md)에 실행 명령·컬럼·결측 정책이 있다.
초기 단일 연도 점검은 [Target 정제 문서](docs/renewable_targets.md)와
[품질 보고서](docs/renewable_target_quality_20261010.md)에 보존한다.

`src/vpp/forecast.py`는 초기 기획의 24시간 수요예측 자리(미구현)이고, 태양광·풍력
발전량 예측은 `src/vpp/generation_forecast/`에서 48시간을 다룬다. 두 기능은 서로
다른 입력·출력 계약을 사용하며, 발전량 예측이 `predict_demand`를 덮어쓰지 않는다.

```text
src/vpp/
  schemas.py                 모듈 간 공통 입출력 타입
  data.py                    시간별 공통 데이터 전처리·로딩
  forecast.py                기존 24시간 수요예측 자리
  generation_forecast/       별도 48시간 태양광·풍력 발전량 예측
    paths.py                 발전량 파이프라인의 공통 저장 경로
    weather_quality.py       일별 GFS 6격자·48시간 품질검사
    dataset.py               발전량 정제 및 GFS 결합 학습마트 생성
    model.py                 기준 모델·LightGBM 학습 및 평가
    tune_lightgbm.py         2023~2024 전용 시간순 LightGBM 튜닝
    audit_labels.py          발전량 결측·이상 출력 진단
    improvement.py           태양광·포트폴리오 개선 피처
    tune_solar_improvements.py 태양광 개선 후보 비교
    wind_hybrid.py           풍력 파워커브·잔차 LightGBM 구성요소
    tune_wind_hybrid.py      풍력 변화량·하이브리드 모델 비교
    tune_wind_improvements.py 풍력 궤적·격자 피처 비교
    tune_oof_ensemble.py     2023 OOF 가중치 학습·2024 평가
    improved_model.py        최종 OOF 앙상블 재학습·추론
    ensemble.py              Weighted Ensemble·Stacking 학습 및 평가
  smp_weather_features.py    GFS 6격자 → SMP 전달용 제주 기상 피처 집계
  market/                    수요·SMP·출력제어 원본 수집 및 통합
  smp_forecast.py            제주 하루전 SMP 48시간 예측 (docs/smp_forecast.md)
  renewable_targets.py       재생에너지 Target 후보 정제
  build_renewable_history.py 과거 재생에너지 자료 통합
  portfolio_data.py          최적화용 포트폴리오 입력 구성
  optimization.py            ESS 운영계획
  validation.py              ESS 계획 제약 검증
  evaluation.py              예측 오차·운영비용 평가
  pipeline.py                모듈 연결 및 CLI

scripts/weather/
  collect_gfs_dayahead_sample.py  GFS 하루 샘플 수집·검증
  collect_gfs_history.py          과거 GFS 48시간 예보 수집·재개

data/
  reference/                 Git에 포함하는 발전소·GFS 격자 기준정보
  raw/
    generation/              발전량 원본 CSV. 직접 수정 금지, Git 제외
    gfs_history_core/        일별 과거 GFS Parquet·수집 로그, Git 제외
    market/                  시장·수요 원본, Git 제외
  processed/
    market/                  시장 정리본 (vpp.market), Git 제외
    forecast/                SMP 예측 결과 (vpp.smp_forecast), Git 제외
    generation_forecast/     정제 발전량·학습마트·모델 지표, Git 제외
    backtest/                ESS 백테스트 실행 결과, Git 제외
  sample/                    화면·재현 확인용 소형 샘플

artifacts/
  models/                    학습 모델 파일, Git 제외
  predictions/               검증·테스트 예측 결과, Git 제외

config/
  ess_v0.yaml                ESS 기본 사양·비용 설정

docs/
  data_contract.md           데이터 소스·시간·단위·전달 형식
  formulation.md             ESS 제약·비용 정의
  generation_asset_validation.md  발전소 21개소 데이터 점검 결과
  generation_forecast_design(11-03).md  발전량 예측 설계
  GITHUB_REPO_HANDOFF.md      발전량 예측 코드 이전 인수인계
  generation_forecast_results.md  발전량 모델 2024 결합 학습·2025 최종 테스트 결과
  smp_weather_features.md     SMP 전달용 제주 기상 피처 계약·생성 방법

api/main.py                  FastAPI 진입점
dashboard/app.py             Streamlit 대시보드
tests/                       모듈별 단위·통합 테스트
.github/                     이슈·PR 템플릿 및 pytest CI
pyproject.toml               패키지·의존성·pytest·ruff 설정
```

## 규칙

- 브랜치 `feat/<모듈>-<내용>` → PR → `main`. `main` 직접 push 금지. PR 전 `pytest`.
- 공통 타입·함수 인자·데이터 형식은 바꾸기 전에 쓰는 사람과 합의.
- 원본 데이터·모델·API 키는 커밋 금지. 키는 `.env`에, 양식은 `.env.example`.
- 원본 데이터는 팀 Drive로 공유. 같은 버전을 각자 `data/raw/`에.

## 발전량·SMP → ESS 입력 검증

공유된 실측 CSV와 예측 파일을 검사하려면 [데이터 인수인계 검증](docs/data_handoff_audit.md)을 따른다.
검증용 예측의 불완전한 48시간 창을 제외하려면 `--incomplete-windows exclude`를 명시한다.
모델·최적화·대시보드의 기존 동작은 변경하지 않으며, 검사 결과는 운영 수익 검증과 구분한다.
[실제 공유 파일 검사 결과](docs/results/forecast_handoff_20261106.md): 완전한 공통 계획 188개.
