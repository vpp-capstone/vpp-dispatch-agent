# 제주 VPP 수요예측·ESS 운영계획

제주 VPP 중개사업자 입장에서 **내일 24시간 전력수요를 1시간 단위로 예측**하고,
그 예측으로 ESS를 어떻게 돌릴지 정한 뒤 실제 비용이 얼마나 줄었는지 본다.
일반 예측 모델부터 만들고, 나중에 운영비용을 학습에 반영하는 모델로 확장.
예측 오차 목표는 5% 이내. 지표(MAPE·nMAE 등)는 팀에서 정함.

이 저장소는 **각 팀이 구현할 함수 자리와 모듈 간 입출력 형식**만 잡아둔 뼈대.
자기 파일의 `TODO`·`NotImplementedError` 자리에 로직을 쓰면 된다.

## 내 자리는 어디?

| 담당             | 파일 · 함수                                                    | 만들 것 → 넘길 것                                            |
| ---------------- | -------------------------------------------------------------- | ------------------------------------------------------------ |
| 데이터&수요예측  | `src/vpp/data.py` — `prepare_hourly`, `load_hourly`            | 수집·정렬·결측 처리 → 시간별 DataFrame                       |
| 데이터&수요예측  | `src/vpp/forecast.py` — `predict_demand`, `train_baseline`     | 특징·모델·학습 → 내일 24개 수요 (`ForecastResponse`)         |
| 최적화           | `src/vpp/optimization.py` — `plan_ess`                         | ESS 제약·비용·솔버 → 충·방전 계획·SOC (`OptimizationResult`) |
| 최적화·대시보드  | `src/vpp/evaluation.py` — `evaluate_forecast`, `evaluate_cost` | 예측 오차 / 실측 기준 운영비용 비교                          |
| 예측·최적화 공동 | `src/vpp/forecast.py` — `train_cost_aware`                     | 위 둘이 끝난 뒤 비용 인지 학습                               |
| 대시보드         | `src/vpp/pipeline.py`, `api/main.py`, `dashboard/app.py`       | 모듈 연결·API·화면                                           |
| 전원             | `src/vpp/schemas.py`, `docs/data_contract.md`                  | 입출력 타입·단위·시각. 바꾸기 전에 쓰는 사람과 합의          |

- `schemas.py`의 최적화·평가 타입은 `status`, `reason`만 있는 빈 틀. 필요한 필드는 담당팀이 추가.
- 함수 인자도 초안. 모델 경로, 발행시각 등 필요하면 합의해서 바꾸면 됨.
- 파일이 커지면 패키지로 쪼개도 됨. import하는 쪽도 같이 고칠 것.

## 순서

1. 데이터 형식·발행시점 합의 (`docs/data_contract.md`)
2. 각자 담당 파일 구현 + 테스트 + 실행 방법
3. 대시보드팀이 `pipeline.py`·API·화면에 연결
4. 일반 예측 → ESS → 비용 평가까지 이어지면 비용 인지 학습

## 설치·실행

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

끄는 건 `Ctrl + C`.
`data/sample/hourly.csv`는 화면 확인용 샘플.
쓸 라이브러리(LightGBM, PuLP 등)는 정한 뒤 `pyproject.toml`의 `dependencies`에 추가.

## 폴더

제주 재생에너지 데이터는 공개 파일 수집 → 과거 파일 정제 → 팀 전달 CSV 생성 순서로 실행한다.
[재생에너지 팀 전달 문서](docs/renewable_history.md)에 실행 명령·컬럼·결측 정책이 있다.
초기 단일 연도 점검은 [Target 정제 문서](docs/renewable_targets.md)와
[품질 보고서](docs/renewable_target_quality_20261010.md)에 보존한다.

수요예측은 기존 `src/vpp/forecast.py`에서 다음 날 24시간을 다룬다. 태양광·풍력
발전량 예측은 `src/vpp/generation_forecast/`에서 48시간을 다루며, 두 기능은 서로
다른 입력·출력 계약을 사용한다.

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
    gfs_history_core/         일별 과거 GFS Parquet·수집 로그, Git 제외
  processed/
    generation_forecast/     정제 발전량 및 태양광·풍력 학습마트, Git 제외
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
  results/                   모델 학습 시 지표·특성 중요도 생성

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
