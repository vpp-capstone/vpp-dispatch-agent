# 배전망 ESS 사업성 모듈 (`vpp.project_economics`)

배전망 ESS 공고 조건으로 "이 선로에 ESS를 넣고 참여할지"를 판단하는 20년 현금흐름 계산기다.
운영 백테스트(48시간 롤링 MILP, 1년)가 내는 1년치 결과를 받아 투자비·지원금·유지비·교체비를 붙인다.

```
vpp.distribution_backtest (1년, 날마다 48시간 MILP)    설정 (config/project_economics_v0.yaml)
  확정 24시간 시간별 결과                               공고 조건, 비용 가정, 배터리 종류,
        |  operating_row_from_hourly()                  PPA 단가, 표준 수익 배분
        v                                                       |
  운영 결과 CSV (case x battery x operation 한 행) -------------+--> project_economics.py
                                                                     |
                               economics_summary.csv (조합별 NPV·IRR·손익분기·참여 판단)
                               economics_cashflow.csv (기준 시나리오 연도별 현금흐름)
```

- 운전(MILP) 설정은 `config/distribution_ess_v0.yaml`(실행 쪽), 사업성 설정은
  `config/project_economics_v0.yaml`(이 모듈)로 나눴다.

## 실행

1단계: 1년 운영 백테스트 → 운영 결과 CSV (`vpp.distribution_backtest`, 2코어 기준 약 30분)

```bash
python -m vpp.distribution_backtest \
  --generation <발전량 예측 평가 파일: artifacts/2025/evaluation/solar_improved_2025_with_actual.parquet> \
  --fcr data/processed/market/fcr_hourly.csv <KPX 예비력 원본 CSV ...> \
  --forecast <SMP v6 예측 CSV: python -m vpp.smp_forecast 출력> \
  --start 2025-01-01 --end 2025-12-31 \
  --out docs/results/distribution_ess_operating_20261115.csv
```

- FCR(`--fcr`, 여러 개): 공공데이터포털 15145101 원본 CSV 또는 `python -m vpp.market.reserve_prices`
  로 원본 엑셀(실시간예비력가격_*.xlsx)을 변환한 시간 CSV(ts, fcr_price_krw_per_kwh). 겹치면 뒤 파일 우선.
  2025년 결과는 `실시간예비력가격_20240601~20250720.xlsx`(변환) + `실시간예비력가격_..._20260630.csv` 로
  1년 전체를 관측값으로 채웠다.
- 태양광: `--generation` 의 포트폴리오 실측(18개소 14.08MW)을 5.7MW로 비례 환산한다
  (`distribution_scenario.scale_solar_to_capacity` 와 같은 비율). 발전량 예측이 없으면
  `--solar <ts, solar_traded_mwh CSV>` 로 KPX 거래량 패턴을 쓸 수 있다 (이때 `forecast_exec` 없음).
- 운전 방식 (`operation`)

  | operation | 계획 | 실행·정산 |
  | --- | --- | --- |
  | `perfect` | 정답 SMP, 실측 발전량 | 계획 그대로, 실제 SMP |
  | `forecast` | SMP v6 예측, 실측 발전량 | 계획 그대로, 실제 SMP |
  | `forecast_exec` | SMP v6 예측, **발전량 예측** | 실측 발전량으로 실행 운전(`vpp.execution`) → 하루전·실시간 이중가격 + 임밸런스 페널티(`vpp.hourly_settlement`) |

  `forecast_exec` 가정: 실행 중 초과 판매 없음(남는 전력은 충전 또는 감축, 선로 상한 유지),
  급전지시량 = 하루전 확정 판매량, 공급가능용량 = 확정 판매량(감발 지시 없음), 자원 설비용량 5.7MW.
  FCR 가격이 없는 시간은 있는 자료의 시간대별 평균으로 채운다. 결과 행에 `deviation_krw`(실시간 편차 정산),
  `shortfall_mwh`, `penalized_mwh`, `imbalance_penalty_krw` 가 붙는다.
- 기본: 설정의 모든 배터리 x 고정 상한 1.5~4.0MW(0.25 간격) + 동적 상한 1.0~2.5MW x 운전 방식 3개.
  `--batteries`, `--fixed-caps`, `--dynamic-caps`, `--existing-pv-mw`, `--operations` 로 바꾼다.
- ESS 용량 비교: `--sizes 2:8,3:12,4:20` (MW:MWh). 행에 `ess_power_mw`·`ess_energy_mwh` 가 붙고
  사업성 계산이 그 용량으로 구축비·지원금을 잡는다.

2단계: 20년 사업성

```bash
python -m vpp.project_economics \
  --config config/project_economics_v0.yaml \
  --operating docs/results/distribution_ess_operating_20261115.csv \
  --out-dir data/processed/economics
```

## 공고 조건 요약 (1차 공고 제2026-337호, 2차 공고 제2026-895호 원문 확인)

| 항목 | 내용 |
| --- | --- |
| 사업 구조 | 선로당 ESS 4MW/20MWh(이 범위 안에서 자율 산정) + 접속대기 태양광 5.7MW 모집, VPP로 시장 참여 |
| 대상 선로 | 배전계통 연계 허용 용량 초과로 접속지연이 생긴 선로. 1차 40개(전남·전북), 2차 34개(강원·경남·경북·충남·광주전남·전북). **제주 선로는 없다** |
| 운전 조건 | 한전이 D-1 09시까지 시간별 ESS 충·방전 허용범위(필요시 재생에너지 출력제어량 포함)를 전달, VPP는 그 안에서 자율적으로 운전계획을 세워 D-1 11~18시에 입찰 |
| 비상시 | 출력제어 미이행·예측 변동·선로 고장 때 한전이 ESS와 재생에너지를 직접 제어 |
| 지원 | 총사업비의 50% 이내, 정격 방전용량 MWh당 최대 5.86억원 |
| 의무 | 배터리 보증 수명 연한(20년)까지 운영, 1년 안에 5.7MW 모집(미달 시 환수 검토) |
| 수익 | 재생에너지 입찰시장 정산금(CP 등), PPA 운영관리 수수료, (신설 예정) 비증설대안(NWAs) |
| 수익 배분 | 자율, 단 "ESS의 공익적 기여도를 고려한" 배분 권장. 접속대기 발전사업자를 차별하거나 이를 통한 수익 창출 금지 |

## 입력 1: 운영 결과 CSV

한 행 = 선로 조건 x 배터리 x 운전 방식, 1년.

| 열 | 뜻 |
| --- | --- |
| `case` | 선로·사업 조건 이름 (예: `cap_1.50MW`) |
| `battery` | 설정 파일 `batteries` 의 키. ESS 없는 기준선은 `none` |
| `operation` | 운전 방식 (`forecast` = SMP v6 예측으로 계획, `perfect` = 정답 SMP) |
| `market_revenue_krw` | 1년 시장수익 = SMP+REC 판매 + 태양광 용량정산금 - 페널티. **열화비는 빼지 않는다**. ESS 용량정산금도 넣지 않는다 |
| `solar_to_vpp_mwh` | VPP가 태양광에서 받은 전력량 = 바로 판매 + ESS 충전. PPA 지급 기준 |
| `ess_charge_mwh`, `ess_discharge_mwh` | 1년 충방전량. 배터리 수명·교체 시점 계산에 쓴다 |
| `days` (선택) | 백테스트 일수. 365가 아니면 1년으로 환산 |

선택 열 (항목별 수익, 다섯 개가 모두 있으면 `standard` 관점도 계산):

| 열 | 뜻 |
| --- | --- |
| `solar_direct_revenue_krw` | 태양광을 바로 판 SMP+REC 수익 |
| `ess_discharge_revenue_krw` | ESS 방전분을 판 SMP+REC 수익 |
| `solar_capacity_payment_krw` | 태양광 용량정산금 = Σ min(바로 판 양, 5.7MW x 3.84%) x RCP |
| `ess_capacity_payment_krw` | ESS 용량정산금 = Σ min(방전량, 4MW x 65.63%) x RCP |
| `imbalance_penalty_krw` | 임밸런스 페널티 (양수 = 낸 돈) |

### 실행 결과에서 행 만들기

시간별 결과(`RollingResult.hourly`: `ts`, `price_krw_per_mwh`, `sell_mwh`, `charge_mwh`,
`discharge_mwh`, 있으면 `rec_krw_per_mwh`·`committed`)를 그대로 넘기면 된다.

```python
from vpp.project_economics import operating_row_from_hourly

row = operating_row_from_hourly(
    plan.hourly,                      # 가격 열은 정산 SMP(실제값)로 둔다
    case="cap_1.50MW", battery="new_2025_bnef", operation="forecast",
    solar_capacity_credit_mw=5.7 * 0.0384,
    ess_capacity_credit_mw=4.0 * 0.6563,
    rcp_krw_per_kwh_h=22.05,          # 제주 RCP
    imbalance_penalty_krw=0.0,        # hourly_settlement 결과가 있으면 그 합계
)
```

- `summarize_distribution_outputs` 의 `sales_revenue_krw` 는 SMP 분만이라 REC가 빠진다.
  위 함수는 `rec_krw_per_mwh` 열을 판매 단가에 더한다.
- ESS 없는 기준선(`battery="none"`)도 같은 함수로 만든다 (충방전 0).
- 배터리별 용량(20MWh x SoH)·효율·열화비는 이 설정의 값을 MILP에 넣어야 결과가 맞는다.
  열화비는 `Battery.throughput_cost_krw_per_mwh()` 로 만든다.

## 입력 2: 설정 (`config/project_economics_v0.yaml`)

| 묶음 | 값 | 근거 |
| --- | --- | --- |
| 사업 | ESS 4MW/20MWh, 태양광 5.7MW, 20년 | 1·2차 공고 |
| 지원금 | 구축비 50%, MWh당 최대 5.86억원 | 1·2차 공고 |
| 금융 | 할인율 7%, 운영비 상승 2%/년 | 가정 |
| 유지비 | O&M 2%/년, 보험·안전검사 0.5%/년 (초기 구축비 대비) | 가정 |
| 교체 | 배터리: 수명 끝나는 해에 배터리 몫만 재구매 (단가 3%/년 하락). PCS: 15년, 구축비의 15% | 가정 (공고 예시 PCS 비중 14.5%) |
| 수익 시나리오 | 1년차 시장수익 대비 연 -2% / 0% / +2% | 가정 |
| PPA | 0, 130, 150, 170원/kWh | 가정 |
| 표준 수익 배분 | VPP 몫: ESS 용량정산금 100%, 태양광 용량정산금 50%, 태양광 에너지 0%, ESS 방전 100%, 페널티 100% | 2차 공고 신청서 작성 예시 (ESS 방전은 표에 없어 가정) |
| ESS 용량정산금 | 넣은 경우·뺀 경우 둘 다 계산 | 2024.9 VPP 내 ESS 별도 인정안 부결, 2차 공고 예시에는 항목 있음 |
| 배터리 | 신품 2021(PNNL), 신품 2025(BNEF, Ember), 국내 예시(57.14억), 사용후(예시) | 각 항목 `source` |

## 관점 (`basis`)

| basis | 비교 대상 | VPP 수익 | 언제 쓰나 |
| --- | --- | --- | --- |
| `standard` | ESS 없으면 접속 못 함 (수익 0) | ESS 방전 수익 + 태양광 용량정산금 50% (+ ESS 용량정산금) - 페널티, PPA 없음 | 공고 조건 그대로. **기본 판단 기준** |
| `total` | ESS 없으면 접속 못 함 (수익 0) | 묶음 전체 시장수익 - PPA x 태양광 수전량 | VPP가 태양광 전력을 PPA로 사는 구조 |
| `incremental` | ESS 없이 상한 안에서 태양광만 판매 (`battery=none` 행) | 시장수익 증가분 - PPA x 늘어난 수전량 | ESS 없이도 접속 가능한 경우 (유연접속 등) |

## 계산식

연도 $t = 0, 1, \dots, N$ ($N = 20$)의 순현금흐름:

$$
CF_0 = -C + S, \qquad
CF_t = R_1 (1+g)^{t-1} - p \cdot Q - (o + i)\, C (1+e)^{t-1} - B_t - P_t \quad (t \ge 1)
$$

- $C$: ESS 구축비 = 단가(USD/kWh) x 환율 x 20,000kWh
- $S$: 지원금 = $\min(0.5\,C,\ 5.86\text{억} \times 20\text{MWh})$
- $R_1$: 관점별 1년차 VPP 수익 (위 표), $g$: 수익 시나리오 변화율
- $p$: PPA 단가(원/kWh), $Q$: PPA를 주는 태양광 수전량(kWh). `standard` 는 $p = 0$
- $o, i$: O&M, 보험·안전검사 비율, $e$: 비용 상승률
- $B_t$: 배터리 교체비 = $C \times$ 배터리 몫 $\times (1 - 0.03)^t$ (수명이 끝나는 해)
- $P_t$: PCS 교체비 = $0.15\,C (1+e)^t$ (15년째)

배터리 수명(년) = 달력 수명과 처리량 수명 중 짧은 쪽,
처리량 수명 = $2 \times$ 사이클 수명 $\times$ 용량 $\times$ SoH $\times$ DoD $\div$ 연 충방전량.

$$
\text{NPV} = \sum_{t=0}^{N} \frac{CF_t}{(1+r)^t}, \quad r = 7\%
$$

IRR은 NPV를 0으로 만드는 $r$. 손익분기 값은 다른 입력은 그대로 두고 그 값만 바꿔 NPV = 0 이 되는 점을
이분법으로 찾는다.

## 출력 열 (`economics_summary.csv`)

| 열 | 뜻 |
| --- | --- |
| `basis` | `standard` / `total` / `incremental` (위 표) |
| `ess_capacity_payment` | ESS 용량정산금을 VPP 수익에 넣었는지 |
| `revenue_scenario`, `ppa_krw_per_kwh` | 시나리오, PPA 단가 (`standard` 는 빈칸) |
| `annual_market_revenue_krw`, `annual_solar_to_vpp_mwh` | 이 관점의 1년차 VPP 수익, PPA 기준 수전량 |
| `solar_owner_gain_krw_per_year` | 태양광 소유자가 1년에 받는 돈 (`standard`: 배분 몫, 그 외: PPA x 수전량) |
| `capex_krw`, `subsidy_krw` | 구축비, 지원금 |
| `battery_life_years`, `battery_replacement_years` | 배터리 수명, 교체 연도 |
| `npv_krw`, `irr`, `payback_years`, `discounted_payback_years` | 투자 지표 |
| `breakeven_subsidy_rate` | NPV = 0 이 되는 지원율 (0 = 지원 없어도 흑자, 빈칸 = 100%여도 적자) |
| `breakeven_capex_krw` | NPV = 0 이 되는 구축비 |
| `min_annual_market_revenue_krw` | NPV = 0 에 필요한 1년차 VPP 수익 |
| `breakeven_ppa_krw_per_kwh` | VPP가 태양광 소유자에게 줄 수 있는 최대 PPA 단가 (빈칸 = 0원이어도 적자 또는 `standard`) |
| `needed_annual_support_krw` | NPV를 0으로 만드는 매년 같은 보상액 (비증설대안 NWAs 보상 등). 흑자면 0 |
| `ess_power_mw`, `ess_energy_mwh` | 계산에 쓴 ESS 용량 |
| `decision` | NPV > 0 이면 `참여 검토`, 아니면 `비참여` |

결과 해석은 [results/distribution_ess_economics_20261115.md](results/distribution_ess_economics_20261115.md).

## 한계 (v0)

- 한전이 주는 시간별 허용범위 대신 고정 상한(낮 8~18시)과 가정한 동적 상한을 쓴다.
- 발전량 예측 오차와 페널티는 `forecast_exec` 에만 있다. 실행 규칙은 판매 약속 우선 규칙
  (`vpp.execution.RuleBasedPolicy`)이라 수익 최적 운전이 아니다.
- 급전지시량·공급가능용량은 실제 급전 기록이 없어 확정 판매량으로 가정했다. 감발 지시(1.5% 허용오차)
  상황은 실시간 가격이 0 이하인 시간에만 생긴다.
- FCR 가격이 없는 시간은 관측분의 시간대별 평균으로 채운다 (2025년은 전부 관측값).
- ESS 용량정산금 실효용량은 용량과 관계없이 4시간 ESS 비율(65.63%)을 쓴다.
- 우리 시장 데이터(SMP, REC, RCP)는 제주인데 공고 대상 선로는 모두 육지다. 육지 선로에 쓰려면 육지
  가격만 바꿔 넣으면 된다.
- 20년치 시간별 운전을 다시 풀지 않는다. 1년 결과에 연 변화율만 곱한다 (용량 감소 미반영).
- 국내 구축비는 공고 신청서 예시값(57.14억)이다. 실제 견적이 생기면 `usd_per_kwh` 를 바꾼다.
- 세금·감가상각, 대출 이자, 비증설대안(NWAs) 보상은 넣지 않았다.
