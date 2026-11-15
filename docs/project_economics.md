# 배전망 ESS 사업성 모듈 (`vpp.project_economics`)

배전망 ESS 공고 조건으로 "이 선로에 ESS를 넣고 참여할지"를 판단하는 20년 현금흐름 계산기다.
운영 백테스트(48시간 롤링 MILP, 1년)가 내는 1년치 결과를 받아 투자비·지원금·유지비·교체비를 붙인다.

```
운영 백테스트 1년 (유준: 계획·실행)          설정 (config/distribution_ess_v0.yaml)
  case x battery x operation 한 행              공고 조건, 비용 가정, 배터리 종류, PPA 단가
              \                                   /
               +--> project_economics.py <-------+
                        |
         economics_summary.csv (조합별 NPV·IRR·손익분기·참여 판단)
         economics_cashflow.csv (기준 시나리오 연도별 현금흐름)
```

## 실행

```bash
python -m vpp.project_economics \
  --config config/distribution_ess_v0.yaml \
  --operating docs/results/distribution_ess_operating_interim_20261115.csv \
  --out-dir data/processed/economics
```

24개 운영 결과(선로 상한 3 x 배터리 4 x 운전 2) 기준으로 몇 초 걸린다.

## 입력 1: 운영 결과 CSV (실행 코드가 만들어 줄 형식)

한 행 = 선로 조건 x 배터리 x 운전 방식, 1년.

| 열 | 뜻 |
| --- | --- |
| `case` | 선로·사업 조건 이름 (예: `cap_1.7MW`) |
| `battery` | 설정 파일 `batteries` 의 키. ESS 없는 기준선은 `none` |
| `operation` | 운전 방식 (`forecast` = SMP v6 예측으로 계획, `perfect` = 정답 SMP) |
| `market_revenue_krw` | 1년 시장수익: SMP+REC 판매 + 용량정산금 (+ 편차·페널티 반영 시 그 값). 열화비는 빼지 않는다 |
| `solar_to_vpp_mwh` | VPP가 태양광에서 받은 전력량 = 바로 판매 + ESS 충전. PPA 지급 기준 |
| `ess_charge_mwh`, `ess_discharge_mwh` | 1년 충방전량. 배터리 수명·교체 시점 계산에 쓴다 |
| `days` (선택) | 백테스트 일수. 365가 아니면 1년으로 환산 |

- 같은 `case`·`operation` 에 `battery=none` 행(ESS 없이 선로 상한 안에서 태양광만 판매)이 있으면
  ESS 증분 관점(`incremental`)을 함께 계산한다. **참여 판단은 이 증분 관점으로 한다.**
- 열화비를 빼지 않는 이유: 열화비는 배터리 구입비를 처리량으로 나눠 붙인 값이라, 구입비를 0년에 넣는
  여기서 또 빼면 두 번 빼게 된다. MILP 운전 판단에 넣을 열화비는
  `Battery.throughput_cost_krw_per_mwh()` 로 같은 설정에서 만든다.

## 입력 2: 설정 (`config/distribution_ess_v0.yaml`)

| 묶음 | 값 | 근거 |
| --- | --- | --- |
| 사업 | ESS 4MW/20MWh, 태양광 5.7MW, 20년 | 2026 2차 공고 (20년 정상 가동 의무) |
| 지원금 | 구축비 50%, MWh당 최대 5.86억원 | 2026 2차 공고 |
| 금융 | 할인율 7%, 운영비 상승 2%/년 | 가정 |
| 유지비 | O&M 2%/년, 보험·안전검사 0.5%/년 (초기 구축비 대비) | 가정 |
| 교체 | 배터리: 수명 끝나는 해에 배터리 몫만 재구매 (단가 3%/년 하락). PCS: 15년, 구축비의 15% | 가정 |
| 수익 시나리오 | 1년차 시장수익 대비 연 -2% / 0% / +2% | 가정 |
| PPA | 0, 130, 150, 170원/kWh | 가정. 0 = VPP가 태양광도 소유하거나 버려지던 전력을 무상으로 받는 경우 |
| 배터리 | 신품 2021(PNNL), 신품 2025(BNEF, Ember), 사용후(예시) | 각 항목 `source` |

배터리 종류는 설정에 한 블록 추가하면 된다 (`usd_per_kwh`, `battery_share`, `soh`, `efficiency`,
`cycle_life`, `calendar_life_years`, `dod`). 사용후 배터리는 이 값만 바꾼 시나리오다.

## 계산식

연도 $t = 0, 1, \dots, N$ ($N = 20$)의 순현금흐름:

$$
CF_0 = -C + S, \qquad
CF_t = R_1 (1+g)^{t-1} - p \cdot Q - (o + i)\, C (1+e)^{t-1} - B_t - P_t \quad (t \ge 1)
$$

- $C$: ESS 구축비 = 단가(USD/kWh) x 환율 x 20,000kWh
- $S$: 지원금 = $\min(0.5\,C,\ 5.86\text{억} \times 20\text{MWh})$
- $R_1$: 1년차 시장수익 (증분 관점이면 ESS 있을 때 - 기준선), $g$: 수익 시나리오 변화율
- $p$: PPA 단가(원/kWh), $Q$: 태양광 수전량(kWh, 증분 관점이면 늘어난 양)
- $o, i$: O&M, 보험·안전검사 비율, $e$: 비용 상승률
- $B_t$: 배터리 교체비 = $C \times$ 배터리 몫 $\times (1 - 0.03)^t$ (수명이 끝나는 해)
- $P_t$: PCS 교체비 = $0.15\,C (1+e)^t$ (15년째)

배터리 수명(년) = 달력 수명과 처리량 수명 중 짧은 쪽,
처리량 수명 = $2 \times$ 사이클 수명 $\times$ 용량 $\times$ SoH $\times$ DoD $\div$ 연 충방전량.

$$
\text{NPV} = \sum_{t=0}^{N} \frac{CF_t}{(1+r)^t}, \quad r = 7\%
$$

IRR은 NPV를 0으로 만드는 $r$. 손익분기 값은 다른 조건을 그대로 두고 NPV = 0 이 되는 값을 이분법으로 찾는다.

## 출력 열 (`economics_summary.csv`)

| 열 | 뜻 |
| --- | --- |
| `basis` | `incremental` (ESS 증분, 참여 판단 기준) / `total` (태양광+ESS 묶음 전체) |
| `revenue_scenario`, `ppa_krw_per_kwh` | 시나리오, PPA 단가 |
| `annual_market_revenue_krw`, `annual_solar_to_vpp_mwh` | 이 관점에서 쓴 1년 시장수익, 태양광 수전량 |
| `solar_owner_gain_krw_per_year` | 태양광 소유자 몫 = PPA x 늘어난 수전량 (증분 관점) |
| `capex_krw`, `subsidy_krw` | 구축비, 지원금 |
| `battery_life_years`, `battery_replacement_years` | 배터리 수명, 교체 연도 |
| `npv_krw`, `irr`, `payback_years`, `discounted_payback_years` | 투자 지표 |
| `breakeven_subsidy_rate` | NPV = 0 이 되는 지원율 (0 = 지원 없어도 흑자, 빈칸 = 100%여도 적자) |
| `breakeven_capex_krw` | NPV = 0 이 되는 구축비 |
| `min_annual_market_revenue_krw` | NPV = 0 에 필요한 1년차 시장수익 |
| `breakeven_ppa_krw_per_kwh` | VPP가 태양광 소유자에게 줄 수 있는 최대 PPA 단가 (빈칸 = 0원이어도 적자) |
| `decision` | NPV > 0 이면 `참여 검토`, 아니면 `비참여` |

결과 해석은 [results/distribution_ess_economics_interim_20261115.md](results/distribution_ess_economics_interim_20261115.md).

## 한계 (v0)

- 20년치 시간별 운전을 다시 풀지 않는다. 1년 결과에 연 변화율만 곱한다 (배터리 열화로 인한 용량 감소 미반영).
- 공고의 "MWh당 5.86억원"은 문구 그대로 MWh 기준으로 읽었다. 실제 단위가 다르면 설정값만 바꾸면 된다.
- 구축비는 해외 단가(USD/kWh)를 환율로 바꾼 값이다. 국내 견적이 생기면 `usd_per_kwh` 를 바꾼다.
- PPA 계약 조건, 태양광 소유 구조는 아직 모른다. 그래서 PPA 단가를 시나리오로 두고 손익분기 단가를 같이 낸다.
- 세금·감가상각, 대출 이자는 넣지 않았다.
