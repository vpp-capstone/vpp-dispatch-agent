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

```text
src/vpp/
  schemas.py        모듈 간 입출력 타입        ← 전원
  data.py           수집·전처리·로딩           ← 데이터&수요예측
  forecast.py       예측·학습                  ← 데이터&수요예측
  optimization.py   ESS 운영계획               ← 최적화
  evaluation.py     예측 오차·운영비용 평가     ← 최적화·대시보드
  pipeline.py       모듈 연결 + CLI            ← 대시보드
api/main.py         FastAPI                    ← 대시보드
dashboard/app.py    Streamlit                  ← 대시보드
data/
  raw/              수집 원본. 직접 수정 금지. Git 제외
  processed/        정제된 시간별 데이터. Git 제외
  sample/           화면 확인용 샘플 CSV
docs/
  data_contract.md  데이터 소스·전달 형식      ← 데이터팀 작성
  formulation.md    ESS 제약·비용 정의         ← 최적화팀 작성
  research_direction.md  참고 논문·방향
tests/              기능 만들면서 각자 추가
.github/            이슈·PR 템플릿, PR마다 pytest 도는 CI
```

## 규칙

- 브랜치 `feat/<모듈>-<내용>` → PR → `main`. `main` 직접 push 금지. PR 전 `pytest`.
- 공통 타입·함수 인자·데이터 형식은 바꾸기 전에 쓰는 사람과 합의.
- 원본 데이터·모델·API 키는 커밋 금지. 키는 `.env`에, 양식은 `.env.example`.
- 원본 데이터는 팀 Drive로 공유. 같은 버전을 각자 `data/raw/`에.
