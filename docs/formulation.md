# ESS 운영계획·비용 평가 (v0)

담당: 최적화·대시보드팀.
구현: `src/vpp/optimization.py`(`EssParams`, `solve_window`, `run_rolling`),
`src/vpp/baselines.py`, `src/vpp/evaluation.py:summarize_backtest`, `src/vpp/validation.py`,
`src/vpp/portfolio_data.py`, CLI `python -m vpp.backtest`. 가정값은 `config/ess_v0.yaml`.
`schemas.py`의 `OptimizationRequest`/`OptimizationResult`는 아직 빈 틀이며 `plan_ess`는 스키마 합의 후
`run_rolling`에 연결한다.

**결과 해석 주의**: v0는 실측 하루전 SMP와 실측 발전량을 그대로 넣는 **완전예측 백테스트**다.
실제 입찰 성과가 아니며, REC·실시간 정산·입찰 편차 페널티는 범위 밖이다.

## 1. 결정 항목 요약

| 결정할 항목 | v0에서 정한 것 |
|---|---|
| 대상 범위 | 제주 중개사업자 가상 포트폴리오: 태양광 18곳(합 14,081 kW) + 풍력 3곳(성산·한경·상명). 네트워크 제약 없이 합산하고 ESS 1대가 전체에 접근한다고 가정. 수력 제외 |
| 입력 | 시간별 발전량 `generation_mwh`(실측 합산), 하루전 SMP `price_krw_per_mwh`(PV 원본 단가 = 제주 SMP, 2024-06-01부터 하루전시장). 둘 다 실측을 예측처럼 사용 |
| ESS | 아래 §4 가정값. 용량 16 MWh, 충·방전 4 MW, SOC 10~90 %, 효율 0.95/0.95, 열화비 0 |
| 목적함수 | 판매수익 − 열화비 (§2). 단위 원 |
| 제약 | 전력균형, SOC 전이·범위, 출력 한도, 동시 충방전 금지, 계통 충전 금지, 선택적 판매 상한 (§2) |
| 기준 방식 | ESS 없음(가격 ≥ 0 전량 판매, 음수면 감축), 규칙 계획(매일 싼 4시간 충전·비싼 4시간 방전, 같은 효율·SOC·이월) |
| 사후 평가 | v0는 완전예측이라 계획 = 실측 적용. 예측 연결 후 `evaluate_cost`에서 고정 계획을 실측에 적용 |
| 해결 방법 | `scipy.optimize.milp`(HiGHS), `mip_rel_gap 1e-8`, 60 s. optimal이 아니면 예외. 결과는 `validation.check_plan`으로 검산하고 실패 시 예외 |

## 2. 수식

변수(시간 t, 단위 MWh): 판매 `q`, 충전 `c`, 방전 `d`, 자발 감축 `r`, 구간 끝 에너지 `E`, 충전모드 `z ∈ {0,1}`
파라미터: 발전 `G[t]` (MWh), 하루전 가격 `p[t]` (원/MWh), 구간 길이 `dt[t]` (h), 열화비 `k` (원/MWh)

```
max  Σ_t p[t]·q[t]  −  k · Σ_t (c[t] + d[t])

G[t] + d[t] = q[t] + c[t] + r[t]                  전력균형 (계통 충전 금지)
E[t] = E[t−1] + η_c·c[t] − d[t]/η_d               SOC 전이, E[−1] = 시작 에너지
SOC_min·Cap ≤ E[t] ≤ SOC_max·Cap
0 ≤ c[t] ≤ min(G[t], P_ch·dt[t]),  c[t] ≤ P_ch·dt[t]·z[t]
0 ≤ d[t] ≤ P_dis·dt[t]·(1 − z[t])                 동시 충방전 금지
q[t] ≥ 0,  0 ≤ r[t] ≤ G[t]
q[t] ≤ C[t]·dt[t]   (선택, 기본 C[t] = ∞ → 제약 없음)
E[마지막] ≥ E_min_final, ≤ E_max_final             (선택, §3 종료 정책)
```

- 에너지 변수이므로 수익 식에 `dt`를 다시 곱하지 않는다. 4 MW는 1시간에 최대 4 MWh.
- 음수·0 가격은 실제 값이며 그대로 둔다 (감축 `r`로 대응).
- `no_ess_same_curtailment_policy_krw = Σ max(p,0)·G` 를 비교용으로 함께 계산한다.

## 3. 풀이 방식 (`run_rolling`, `policy`)

| policy | 창 | 확정 | 종료 제약 |
|---|---|---|---|
| `rolling48` (기본) | 매일 D, D+1 48시간 | 앞 24시간만 | **백테스트 마지막 창**에서 `E[마지막] ≥ 기간 시작 에너지` |
| `free` | 24시간 | 전부 | 없음 (Codex 일별 풀이와 동일) |
| `equal_initial` | 24시간 | 전부 | `E[창 끝] = E[창 시작]` |

- 확정된 24시간 끝 에너지를 다음 창 시작 에너지로 넘긴다. 데이터 끝에서 48시간이 안 되면 남은 시간만 푼다.
- 기간 끝 잔량 제약은 처음 들어 있던 에너지를 팔아 생긴 돈이 ESS 효과로 섞이지 않게 하기 위함이다.
- 결측으로 제외한 날의 앞뒤는 연속된 것처럼 이어 붙여 푼다(잔량 이월·48시간 창이 제외된 날을 건너뜀).
- 2025-04 `free` 결과는 Codex `ess-v0` 결과와 일치(1,178,887,507.81원, 종료 1.6 MWh)한다.

## 4. 가정값 (`config/ess_v0.yaml`, `status: assumption_not_equipment_spec`)

| 항목 | 값 | 비고 |
|---|---|---|
| 용량 | 16 MWh | 설비 근거 없음. 민감도·설비 선정은 후속 |
| 충전·방전 출력 | 각 4 MW | |
| SOC 범위 | 10 ~ 90 % (1.6 ~ 14.4 MWh) | |
| 시작 SOC | 50 % (8 MWh) | 백테스트 기간 첫 시작에만. 이후 이월 |
| 충전·방전 효율 | 0.95 / 0.95 | 왕복 90.25 % |
| 열화비 | 0 원/MWh | 미반영. 실제 비용이 0이라는 뜻 아님 |
| 계통 충전 | 금지 | v0는 금지만 지원 |

## 5. 데이터 규칙 (`portfolio_data.build_portfolio_hourly`)

원본 4개는 `--raw-dir`로 받으며 레포에 복사하지 않는다 (기본 `data/raw/portfolio/`).

| 파일 | 인코딩 | 형태 | 단위·처리 |
|---|---|---|---|
| `제주#1_기상정보 융합 태양광 발전 시계열 데이터.csv` | utf-8-sig | long: 거래일, 시간(1~24), 발전기코드, 계량전력량, 단가 | kWh, 18곳 합산. 단가(원/kWh)는 같은 시각 18곳 동일 → 시간별 1개. 불일치·발전소 수 부족 시 그 날 제외 |
| `한국남부발전(주)_성산풍력발전실적_20260331.csv` | cp949 | wide: 년월일, 단계(1·2), 1~24 | kWh, 단계 1+2 합산. `-`·공백 → NaN |
| `한국남부발전(주)_한경풍력발전실적_20260331.csv` | cp949 | 동일 | 동일. 2023-01-30 1단계 중복 2행은 값이 달라 **자동 선택 없이 그 날 제외**(값이 같은 중복은 하나만 유지) |
| `한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv` | cp949 | wide: 기준일, 호기, 1시~24시 | **2025-01-31까지 Wh → ÷1000, 2025-02-01부터 kWh 그대로**(두 구간 모두 최대 ≈ 20.1 MWh, 21 MW 설비와 일치). 호기가 여럿이면 합산 |

- 시각: 원본 hour h = (h−1)시~h시. `ts`는 구간 시작 KST (1시 → 00:00). `docs/data_contract.md`와 같은 규칙.
- 설비 근거(플래그용): 성산 1단계 12 MW·2단계 8 MW, 한경 1단계 6 MW·2단계 15 MW, 상명 21 MW, 태양광 18곳 합 14,081 kW.
  용량 초과 시간은 `*_over_capacity` 플래그만 남기고, `--clip-to-capacity`일 때만 클리핑.
- 결측·중복·이상치는 보간하지 않는다. 결측이 있는 날은 제외하고 `validation.json`의 `excluded_days`에 사유와 함께 기록.
- 가격 대조: `data/processed/market/market_hourly.csv`의 `smp_da`가 있으면 시간별 비교해 불일치 시간 수를 기록. 없으면 건너뜀.
- 기본 기간 2024-06-01 ~ 2025-12-31 (제주 시범사업 이후). 2025-04 한 달 실행으로 Codex와 비교 가능.

## 6. 평가·검증

- `evaluation.summarize_backtest`: 판매수익, 열화비, 순수익, 기준 대비 증가액·증가율(ESS 없음, 규칙 계획), 일평균 사이클(방전량 ÷ 사용가능용량 ÷ 일수), 감축량, 기간 시작·끝 에너지.
- `validation.check_plan`: 풀이 상태, 전력균형, SOC 재계산 일치, SOC 범위, 충전·방전 상한, 판매·감축 범위, (판매 상한), 동시 충방전 없음, 종료 에너지, 목적함수 재계산 일치, 롤링 에너지 연속. 위반은 예외 대신 `{item, passed, value}` 목록으로 반환하고 `optimization`이 실패 시 예외를 던진다.
- `validation.check_properties`: MILP ≥ ESS 없음, MILP ≥ 규칙 계획, (크기 여러 개 실행 시) 용량 증가 시 수익 비감소.
- 출력: `data/processed/backtest/<실행ID>/hourly_plan.csv`, `daily_summary.csv`, `baselines_hourly.csv`, `summary.json`, `validation.json`(가정값·한계·데이터 검사·제약 검산 포함). 대시보드 "ESS 운영계획"·"검증" 탭에서 본다.

## 7. 범위 밖 (후속)

ESS 후보 크기 비교, 효율·열화비 민감도, REC 항, 실시간 정산·입찰 편차 페널티, 출력제어 상한 데이터 연결,
예측 연결 후 실측 적용 평가(`evaluate_cost`), `schemas.py`·`pipeline.py`·API 연결.
