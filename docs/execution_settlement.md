# ESS 실행 운전과 시간별 사후 정산

1~5절은 기존 실행·백테스트의 동작이다. 6절부터는 1단계에서 추가한 **독립 시간 정산 API/CLI**를
설명한다. 새 기능은 `smp_da`와 `smp_rt_mean`을 분리하고 현행 임밸런스 페널티를 1시간으로 근사한다.
기존 `settlement.py`와 `execution_v0.yaml`은 변경하지 않았으며, 최적화·실행 정책·백테스트·
운영 대시보드에는 아직 연결하지 않았다. 별도 조회 화면은 저장된 1단계 결과만 읽는다.
두 방식 모두 실제 청구 정산액을 재현하는 기능이 아니며, 적용한 가정과 계산 가능 여부를 기록한다.

## 1. 입력과 원칙

- 예측 발전량과 예측 SMP로 만든 확정 판매계획은 실행 단계에서 변경하지 않는다.
- 실측 발전량에 맞춰 ESS 충전·방전만 시간순으로 조정한다.
- 미래 시각의 실측 발전량이나 실측 가격은 운전 결정에 사용하지 않는다.
- 전날 실행 종료 SOC를 다음 날 실행과 계획의 시작 SOC로 이월한다. 기간이 끊기면 설정 SOC로 다시 시작한다.

구현은 `src/vpp/execution.py`, 일별 SOC 연결은 `src/vpp/execution_rolling.py`, 정산은
`src/vpp/settlement.py`에 있다.

## 2. 실행 운전 규칙

발전량이 확정 판매량보다 부족하면 계획 충전을 먼저 줄이고, 그다음 SOC 하한 안에서 추가 방전한다.
그래도 부족한 양은 `shortfall_mwh`로 남긴다.

잉여 전력은 기본적으로 다음 순서로 처리한다.

1. 계획 방전 축소
2. 초과 판매
3. 추가 충전
4. 출력 감축

초과 판매는 계획 시점 판매 가치인 예측 SMP와 REC 가산의 합이 0 이상일 때만 허용한다. 이 규칙은
설명 가능한 실행 정책이며 수익 최적 운전을 보장하지 않는다.

## 3. 기존 API의 실현 수익 계산

현재 계산식은 다음과 같다.

```text
실현 순수익
= 확정 판매 정산액
+ 초과·부족 편차 정산액
+ REC 수익
- 실제 충·방전에 따른 열화비
```

- 확정 판매 정산액: `확정 판매량 × 하루전 정산 가격`
- 편차 정산액: `초과량 × 초과 단가 - 부족량 × 부족 단가 - 부족 벌칙`
- REC 수익: 실제 공급량에 한 번 적용
- 열화비: 실제 실행 충전량과 방전량으로 계산

## 4. 기존 API의 정산 가정과 한계

시장 데이터 스키마에는 `smp_da`, `smp_rt_mean`이 있지만, 기존 `settle` 호출은 두 가격을 분리해
전달하지 않아 둘 다 해당 시각의 실측 SMP로 대체한다.
초과·부족 가격 배수는 1.0, 부족 추가 벌칙은 0원/MWh다. 따라서 결과의 상태는
`experiment_assumption_not_market_rule`, 가격 기준은 `assumed_actual_smp`로 기록한다.

정산 가격이 한 시간이라도 없으면 해당 일자의 실현 수익은 계산하지 않으며 예측 가격으로 채우지 않는다.
발전량 예측 정산금과 용량정산금은 현재 실현 수익에 포함하지 않는다. 초기 SOC에 저장돼 있던 전력의
REC 자격도 별도로 추적하지 않는다.

## 5. 결과 해석

대시보드의 세 금액은 서로 다른 비교값이다.

1. 예측으로 세운 계획의 예상 수익
2. 확정 계획을 실측 발전량에 적용한 가정 기반 실행 수익
3. 실측을 미리 알고 세운 완전예측 비교 계획의 수익

두 번째 값만 실행 결과지만 실제 정산 가격을 사용한 수익은 아니다. 세 값을 합산하거나 서로 대신 사용하지
않는다. 설정과 결과 표시는 [운영 대시보드 문서](operator_dashboard.md)를 따른다.

## 6. 1단계: 1시간 이중가격 정산과 페널티

구현 파일:

- `src/vpp/hourly_settlement.py`: `settle_hourly`, 일별·전체 집계, 독립 CLI.
- `src/vpp/settlement_inputs.py`: 시간 정렬, 입력 검증, 설정과 출처.
- `src/vpp/imbalance.py`: 조건별 허용오차와 페널티 단가.
- `src/vpp/market/reserve_prices.py`: 원본 예비력 엑셀에서 FCR 시간 평균 생성.
- `dashboard/settlement_review.py`: 저장된 시간별 정산 결과 조회.
- `config/settlement_jeju_hourly.yaml`: 현행 규칙 버전과 손실계수·REC 가정.

### 정산 식과 적용 범위

```text
하루전 정산액 = committed_sell_mwh × (smp_da × 하루전 손실계수 × 1,000)
판매 편차 = supply_mwh − committed_sell_mwh
편차 정산액 = 판매 편차 × (smp_rt_mean × 실시간 손실계수 × 1,000)
에너지 수익 = 하루전 정산액 + 편차 정산액
페널티 전 수익 = 에너지 수익 + REC − 실행 열화비
실현 순수익 = 페널티 전 수익 − 임밸런스 페널티
```

가격 원/kWh를 원/MWh로 **한 번만 1,000배** 한다. 음수 가격에도 편차의 부호를 유지한다.
실제 공급량 전체에 SMP를 한 번 더 곱해 더하지 않는다. `supply_mwh`를 단일 모델 자원의
계량 공급량으로 간주한다. 실제 등록 자원 경계와 계량 범위 일치는 별도 확인 대상이다.

REC 기본값은 `supplied_mwh_once`로, 기존 실행의 `rec_krw_per_mwh`를 실제 공급량에 한 번
적용한다. 기존 `RecParams`가 적용한 월별/고정 단가를 그대로 쓰며 새 단가를 만들지 않는다.
REC 열이 없으면 오류이며 의도적으로 제외하려면 `excluded`를 지정한다. REC 자격·초기 SOC의
자격은 판정하지 않는다. 열화비는 기존 `EssParams` 모델로 실제 충·방전량에서 계산한다.
CLI에서 `--ess`를 생략하면 실행 CSV와 같은 폴더의 `summary.json`에 저장된 `ess`를 재사용한다.
이 정보도 없으면 `--ess`를 요청하며, 열화비 0 설정으로 자동 대체하지 않는다. 용량정산금·예측정산금
등은 포함하지 않는다. 예제에서만 `ess_v0.yaml`의 열화비 0 가정을 명시적으로 선택한다.

### 임밸런스: 판매 편차와 별개의 급전 편차

기준은 전력시장운영규칙 **별표 33, 인쇄면 949~950**의 임밸런스 조항이다.
[규칙 원문(2026년 5월)](https://kpx.or.kr/boardDownload.es?bid=0030&list_no=77395&seq=5)과
[6%·1.5% 적용 공지](https://new.kpx.or.kr/board.es?act=view&bid=0042&list_no=76974&mid=a11201000000)를
근거로 `jeju_2026_04` 버전을 고정했다. 시행 거래일은 **2026-04-01**이다. 이전 기간에도 이 버전을
사용하지만 `current_rule_counterfactual`로 표시하며 당시 정산을 재현했다고 해석하지 않는다.

`RT_MP = smp_rt_mean × 실시간 손실계수`이고, 아래 표의 가격 단위는 원/kWh다.

| 조건 | 허용오차율 | 급전 목표 초과 시 페널티 단가 | 급전 목표 미달 시 단가 |
|---|---:|---|---|
| `RT_MP <= 0` | 1.5% | `max(-입찰하한가격, 0) × 2.5 + FCR가격` | 0 |
| `RT_MP > 0` 및 `SET_POINT < RA` | 1.5% | `max(RT_MP - 자원 최소입찰가격, 0) + FCR가격` | 0 |
| `RT_MP > 0` 및 `SET_POINT >= RA` | 6% | FCR가격 | FCR가격 |

```text
급전 편차 MWh = supply_mwh − set_point_mw × 1h
허용 편차 MWh = resource_capacity_mw × 허용오차율 × 1h
허용범위 밖 물량 MWh = max(abs(급전 편차) − 허용 편차, 0)
페널티 비용 원 = 허용범위 밖 물량 × 위 단가 × 1,000
```

- 6%·1.5%는 수익에 곱하는 벌금률이 아니라 **주요 발전자원의 설비용량 기준 허용오차율**이다.
  ESS MWh 용량이나 확정 판매량을 기준으로 삼지 않는다.
- 단순한 급전 발생 여부 플래그만으로 결정하지 않는다. RT 가격과 급전 목표·변경 공급가능용량을
  함께 비교한다. 급전 목표를 판매계획에서 자동 생성하지 않는다.
- 단가 0인 미달 면제 분기는 허용범위 밖 물량이 있어도 비용 0이다. `penalized_mwh`는 이 물량을
  기록하므로 실제 차감 여부는 `imbalance_penalty_krw`로 확인한다.
- 모든 입력·출력은 1시간이다. 원문의 구간별 계산을 시간 평균 가격으로 근사하므로 `0.25h`를
  다시 곱하거나 4배 하지 않는다. 시간 안에서 가격 부호·분기가 바뀌는 효과는 재현하지 않는다.

### 입력 CSV

세 입력의 `ts`는 **시간 시작**이다. 타임존 없는 값은 KST로 해석하고, 타임존이 있으면 KST로
변환한다. 행 순서가 아닌 시각으로 결합한다. 중복 시각, 15분 시각, `dt_hours != 1`, 다른
`resource_id`의 혼합은 오류로 처리한다. 실행 자료는 하나의 자원·시나리오만 전달한다.
여러 자원의 편차를 합쳐 상계해서는 안 된다.

| 입력 | 필드 | 단위/의미 |
|---|---|---|
| 실행 | `ts`, `dt_hours` | KST 시간 시작, 반드시 1 |
| 실행 | `committed_sell_mwh`, `supply_mwh` | 확정 판매량, 실제 공급량 MWh |
| 실행 | `charge_mwh`, `discharge_mwh` | 실제 충·방전량 MWh, 열화비 계산용 |
| 실행 | `rec_krw_per_mwh` | 기본 모드에서 필수. 기존 실행·월별 REC 단가 재사용 |
| 시장 | `ts`, `smp_da`, `smp_rt_mean` | `market_hourly.csv`의 시간 가격, 원/kWh |
| 급전 | `resource_capacity_mw` | 주요 발전자원의 설비용량 ICDM, 양수 MW |
| 급전 | `set_point_mw` | EMS 60분 평균 급전 목표 SET_POINT, MW |
| 급전 | `available_capacity_mw` | 변경 공급가능용량 RA의 시간 평균 출력 환산값, MW |
| FCR (`--fcr`) | `ts`, `fcr_price_krw_per_kwh` | 시간 단위 실시간 주파수제어예비력 가격 |
| 급전 | `min_offer_price_krw_per_kwh` | 해당 자원·시간의 구간별 입찰가격 중 최소값 |
| 급전 | `offer_floor_krw_per_kwh` | 해당 시간에 적용할 시장 입찰하한가격 |
| 급전 | `dispatch_source`, `dispatch_note` | `observed`/`scenario`/`unknown`, 출처·가정 설명 |

관측·시나리오 출처를 지정하면 `dispatch_note`가 필수다. 급전 파일의 출처 표시는 그 행의
급전·용량·입찰 입력에 대한 것으로, 일부라도 가정이면 `scenario`로 표시하고 상세히 적는다.
출처를 생략하면 `unknown`으로 기록하며 관측 자료라고 추정하지 않는다. FCR은 `--fcr`로
독립 입력하고 `fcr_price_source`로 출처를 남긴다. 기존 예제처럼 급전 CSV에 FCR을 포함하는
입력도 지원한다. 두 파일에 같은 시각의 다른 FCR 가격이 있으면 오류다. `--fcr`를 지정하면
해당 파일의 누락 시간은 급전 CSV 값으로 채우지 않는다.
손실계수는 설정에서 지정하고 출처와 설명을 결과에 남긴다. 기본 1.0은 명시적인 가정이다.

없는 가격·급전 시각이나 필수 입력은 보간·전방 채움·예측값 대체를 하지 않는다. 에너지 수익만
계산할 수 있으면 해당 금액은 남기되 페널티와 차감 후 수익은 NaN/JSON null로 둔다. 일반 6% 분기의
과금에는 FCR만 필요하다. 양수 RT 가격에서 감발 조건의 초과분에는 FCR·자원 최소입찰가격,
RT 가격이 0 이하인 초과분에는 FCR·시장 입찰하한가격이 필요하다.
허용범위 이내 또는 단가 0인 미달 면제가 확인된 경우에는 관련 없는 단가 없이도 페널티 0을
확정할 수 있다. 비수치·무한대·음수 물리량은 결측으로 숨기지 않고 오류 처리한다.

### 실행 및 확인

저장소 루트에서 다음을 실행한다. 예제의 **시장·실행·급전 값은 모두 합성 검산 자료**다.

```bash
.venv/bin/python -m vpp.hourly_settlement \
  --execution data/sample/settlement_hourly/execution.csv \
  --market data/sample/settlement_hourly/market.csv \
  --dispatch data/sample/settlement_hourly/dispatch.csv \
  --config config/settlement_jeju_hourly.yaml \
  --ess config/ess_v0.yaml \
  --outdir data/processed/settlement/stage1_sample

.venv/bin/python -m pytest -q tests/test_hourly_settlement.py tests/test_settlement.py
```

출력 폴더는 비어 있어야 한다. 재실행 시 새 폴더명을 사용한다. 실제 실행 CSV에 여러 시나리오가
있으면 `--scenario 이름`으로 하나를 고른다. `--dispatch`를 생략해도 에너지 수익은 확인할 수
있지만 페널티와 최종 순수익은 계산 불가로 기록된다.

- `hourly_settlement.csv`: 두 가격, 판매·급전 편차, 조건 분기, 허용오차, 적용 단가, 페널티,
  차감 전후 수익, 입력 출처, `settlement_missing_inputs`/`imbalance_missing_inputs`.
- `daily_settlement.csv`: KST 일별 합계와 `full_day_coverage`. 24시간이 없으면 False다.
- `summary.json`: 설정·입력 경로·ESS 모델, 합계, FCR 가격 누락 시간 수,
  페널티 발생/0/누락 시간 수, 출처별 시간 수.

한 시간이라도 항목 계산이 불가능하면 그 항목의 일별·전체 합계는 null/빈칸이다. 누락값을
제외한 부분합으로 표시하지 않는다. `computable`은 **제공된 실행 행**의 계산 가능 여부이며
기간 전체 완전성을 보증하지 않는다. 빠진 실행 시간은
`missing_execution_hours_between_start_end`, 부분 일자는 `full_day_coverage`로 확인한다.
예제 기대값과 검산 순서는 `data/sample/settlement_hourly/README.md`에 있다.

### 완료 범위와 후속 작업

1단계의 완료 기준은 **가격 연결, 독립 정산 계산기, 기존 합성 예제 검산, 결과 조회**다.
실제 급전·입찰 자료 수집, 새 급전 시나리오 생성, 판매·충방전 의사결정 변경은 포함하지 않는다.
기존 7시간 검산 자료는 계산식 확인용이며 후속 운영 급전 시나리오와 구분한다.

`optimization.py`, `execution.py`의 의사결정은 바뀌지 않았다. 최적화·실행에 페널티를 반영하는
것은 2단계, 운영 대시보드 자동 계산 연결과 ESS 전후 비교·전체 기간 검증은 3단계다.
현재 조회 화면은 정산을 실행하거나 급전값을 생성하지 않는다.

### 기존 자료 재사용과 미계산 항목

기존 실행의 `executed` 행 4,512시간을 사용한다. `--scenario executed`는 기존 결과 선택이며
새 급전 시나리오를 생성하는 옵션이 아니다. 실행량·REC 단가와 실행 당시 ESS 설정을 재사용한다.

| 점검 항목 | 결과 |
|---|---|
| 기존 실행 범위 | 2024-06-01~12-30 중 188일, 4,512시간 |
| 하루전·실시간·FCR 가격 연결 | 4,512시간 모두 매칭, 가격 누락 0 |
| 기존 계획·실행량·REC·열화비 | 보존 |
| 기간 중 빠진 실행 시간 | 원래 실행 결과에서 제외된 600시간; 정산에서 생성·보간하지 않음 |
| 기존 실행 기록의 페널티·차감 후 수익 | 급전·입찰 입력 없이 미계산으로 유지; 0원 처리하지 않음 |

기존 `plant_location_master_21.csv`에 태양광 14.08122MW, 풍력 62MW가 있으므로 설비용량을
다시 수집할 필요는 없다. 후속 실험에서 정산 대상 자원 범위를 정한 뒤 해당 용량을 연결한다.
실제 급전 기록을 확보하거나 새 시나리오를 만드는 것은 1단계 검산의 완료 조건이 아니다.

### 시장 가격 파일의 가공 이력과 4구간 검사

`market_hourly.csv`는 이미 시간 단위로 가공된 파일이다. 전달받은 `market_hourly (1).csv`를
바이트 변경 없이 복사했으며, 이 복사 과정과 그 이전의 15분→1시간 집계를 구분해
`market_hourly.source.json`에 기록한다. 도구 이름은 데이터 출처로 사용하지 않는다.

다운로드 폴더의 `jeju_rt_15min.csv`를 이용해 `smp_rt_mean/min/max`를 재계산했다.
기존 `src/vpp/market/build.py`와 같은 시간 시작 기준 집계·소수 둘째 자리 반올림을 적용하면
현재 시장 파일의 세 열과 일치한다. 15분 입력의 출처 표시는 `api` 79,584행과 `xlsx` 11,904행이며,
수집 모듈에 명시된 출처는 각각 KPX 제주시범사업정보 API와 제주 실시간 SMP 엑셀이다.
이번 확인은 이 병합된 15분 파일 이후의 집계 검증이며, 기관 원본 응답·엑셀까지 역추적한 것은 아니다.

- 15분 입력 91,488행, 2024-03-01~2026-10-09: 중복·15분 시각 이탈·가격 결측 없음.
- 가격이 있는 22,872시간 모두 00·15·30·45분의 유효 가격 4개를 포함한다.
- 기존 실행에 쓰인 4,512시간도 모두 4개이며, 현재 데이터에는 부분 평균이 없다.
- 시장 파일의 나머지 36,576시간은 이 RT 자료 범위 밖이며 RT 가격을 결측으로 유지한다.

이 검사는 **현재 데이터의 완전성**을 확인한다. 기존 `build.py`는 유효 값이 1~3개일 때도
부분 평균을 내므로, 향후 새 자료를 가공할 때는 별도로 4개 검사와 결측 처리가 필요하다.
이번 범위에서는 기존 가공 코드·시장 CSV·정산 입력 스키마를 변경하지 않고 검증 결과를
출처 JSON에 남긴다. FCR 가공은 이미 4개 미만이면 결측으로 처리한다.

### FCR 시간 평균 변환과 실제 실행 연결

입력은 `실시간예비력가격_20240601~20250720.xlsx`다. 원본을
`data/raw/market/reserve_prices/`에 그대로 보존했다. 날짜가 예비력 3행의 가운데에 있으므로
3행 묶음 안에서 날짜를 확인하고 `주파수제어예비력` 행만 읽는다. 날짜를 단순 전방 채움하지 않는다.
01시의 4구간 평균은 KST 00:00에, 24시 평균은 같은 날짜 23:00에 대응한다.
네 값 중 하나라도 결측이면 시간 가격도 결측으로 남긴다. 평균에 별도 반올림은 하지 않는다.

```bash
.venv/bin/python -m vpp.market.reserve_prices \
  --input 'data/raw/market/reserve_prices/실시간예비력가격_20240601~20250720.xlsx' \
  --out data/processed/market/fcr_hourly.csv

.venv/bin/python -m vpp.hourly_settlement \
  --execution data/processed/backtest/exec_2024_h2/hourly_execution.csv \
  --scenario executed \
  --market data/processed/market/market_hourly.csv \
  --fcr data/processed/market/fcr_hourly.csv \
  --outdir data/processed/settlement/stage1_existing_execution_fcr
```

정산 출력 폴더가 이미 있으면 새 이름을 지정한다. 원본 415일을 9,960시간으로 변환했으며,
기존 실행 4,512시간에 FCR 가격이 모두 연결된다. `fcr_hourly.source.json`에 원본 경로·해시·
집계법·단위를 남겼다. 원문 별표 33의 FCR 단위는 원/kW-h이며, 1시간 모델의 기존 필드
`fcr_price_krw_per_kwh`에 같은 수치를 사용한다. 정산 입력에서 원/MWh로 한 번만 1,000배 한다.
`market_hourly.csv`, 판매량, 공급량, 충방전량, REC·열화비는 변경하지 않는다.

출력 폴더의 `fcr_validation.json`은 원본 셀에서 독립적으로 다시 계산한 9,960시간 평균,
4,512시간 매칭, 기존 정산 항목 보존과 단위 검산 결과다. 실제 FCR 13.15를 넣은 손계산에서는
일반 6% 분기 허용범위 밖 물량 0.25MWh의 비용이 `0.25 × 13.15 × 1,000 = 3,287.50원`이다.
이 계산은 검산용이며 운영 급전 시나리오를 생성한 것이 아니다. 실제 실행의 급전 입력은 여전히
없으므로 페널티·최종 순수익은 계산 불가로 남고, `fcr_price_missing_hours`만 0이 된다.

### 대시보드에서 확인

기존 `dashboard/app.py`의 운영 화면은 이전 백테스트 정산을 사용한다. 새 결과는 별도 조회
화면에서 확인하며, 이 화면은 최적화·실행·자동 파이프라인을 호출하지 않는다.

```bash
.venv/bin/python -m streamlit run dashboard/settlement_review.py --server.port 8502
```

기본으로 `stage1_existing_execution_fcr` 결과를 읽는다. 사이드바에서 다른 정산 결과 폴더와
조회 기간을 선택할 수 있다. FCR 연결 시간 수, 하루전·실시간·FCR 가격, 판매량과 공급량,
페널티 차감 전 수익, 누락 입력을 확인하고 선택 기간 CSV를 내려받을 수 있다.
`선택한 시간의 적용 가격과 계산`에서 한 시간을 고르면 판매·편차 정산식과 6%·1.5% 판정,
급전 목표·공급가능용량, 허용범위 밖 물량과 조건별 페널티 단가 계산을 볼 수 있다.
급전 입력이 없는 페널티·최종 순수익은 **미계산**으로 표시한다. 같은 실행 결과의 가격 변화와
편차 정산을 살펴보는 용도이며, ESS 최적 운전 효과나 공식 정산 수익을 입증하는 화면은 아니다.

사이드바의 **조건별 검산 예제**는 기존 7시간 합성 예제 결과를 읽는다. 아래 명령으로 생성한다
(이미 생성했으면 그대로 사용). 새 운영 급전 시나리오를 만드는 기능은 아니다.

```bash
.venv/bin/python -m vpp.hourly_settlement \
  --execution data/sample/settlement_hourly/execution.csv \
  --market data/sample/settlement_hourly/market.csv \
  --dispatch data/sample/settlement_hourly/dispatch.csv \
  --ess config/ess_v0.yaml \
  --outdir data/processed/settlement/stage1_sample_view
```

`2024-06-01 01:00`은 양수 가격·감발 조건의 1.5%(페널티 40,000원), `03:00`은 일반
6%(1,000원), `04:00`은 실시간 가격 -20원/kWh에서의 1.5%(51,250원) 예제다.
예제의 가격은 모두 검산용 가상 값이며, 시장 가격 자체를 생성하거나 예측하는 화면은 아니다.

원본·변환 데이터・출처 JSON・정산 결과는 기존 Git 제외 정책을 따른다.
데이터를 공유할 때는 해당 CSV와 `.source.json`을 함께 전달한다.
