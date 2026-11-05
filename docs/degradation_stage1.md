# ESS 1단계 열화모델

## 범위

v0.1은 현재 데이터로 추정할 수 있는 가장 작은 모델인 **고정 처리량 열화비**를
사용한다. 시간·온도·SOC·DoD에 따라 열화율이 달라지는 경로 의존 모델은 2단계 이후로
제외한다. 따라서 v0.1의 비용은 셀 열화의 실시간 상태 추정치가 아니라, 교체비 proxy를
기대 처리량에 배분한 MILP 목적함수 비용이다.

## 정제한 근거와 가정

PNNL/DOE의 [Energy Storage Grand Challenge Cost and Performance Assessment 2022]
(PNNL-33283)에서 다음 공개자료를 사용했다.

| 항목 | 값 | 의미 |
|---|---:|---|
| DC Storage Block 비용 | 182.27 USD/kWh | 2021년 LFP 1 MW·4시간 기준의 DC 저장블록 비용. 전체 EPC나 실제 교체비가 아닌 proxy |
| 기준 DoD | 80% | LFP cycle-life 표의 기준 DoD |
| cycle life | 2,400 cycles | 해당 기준의 공개 수명값 |
| v0 용량 | 16 MWh | 저장소의 설비 확인값이 아닌 v0 가정 |
| 환율 | 1,350.36 KRW/USD | 2026-11-03 환산값; 민감도는 별도 CSV에 저장 |

원자료 PDF와 출처 메타데이터는 `data/raw/degradation/`에, 계산에 사용한 정제값은
`data/processed/degradation/stage1_basis.csv`와 `stage1_cost_scenarios.csv`에 저장했다.

## 산식

현재 스케줄러의 처리량 단위는 AC-side `charge_mwh + discharge_mwh`이다.

```text
replacement_cost_proxy_krw = 182.27 USD/kWh × 1,350.36 KRW/USD × 16,000 kWh
                           = 3,938,081,875.20 KRW

expected_throughput_mwh = 2 × 2,400 cycles × 16 MWh × 0.80 DoD
                        = 61,440 MWh

throughput_cost = replacement_cost_proxy / expected_throughput
                = 64,096.38 KRW/MWh
```

그러므로 v0.1은 충전량과 방전량 각각에 64,096.38 원/MWh를 목적함수 비용으로
부과한다. 비용이 0인 `config/ess_v0.yaml`은 v0, 비용이 적용된
`config/ess_v0_1.yaml`은 v0.1이다.

이 값은 실제 적용 ESS의 확정 열화비가 아니라 공개자료 기반 시나리오 proxy이다.
또한 원자료의 cycle-life throughput 정의와 현재 코드의 `charge_plus_discharge`
convention 사이의 변환은 후속 검토 대상이다.

## 커스텀 모델 확장점

`src/vpp/degradation.py`의 `DegradationModel`은 MILP 충전·방전 계수와 사후 비용
계산을 분리한다. 기본 이름은 `none`과 `fixed_throughput`이며, 커스텀 구현은
`register_degradation_model(name, factory)`로 등록해 같은 `EssParams`와 스케줄러에
연결할 수 있다. factory의 closure로 DoD 구간 등 추가 파라미터를 캡처할 수 있다.
온도·누적 사이클처럼 비선형 또는 경로 의존적인 모델은 별도의 선형화·상태변수 설계가
필요하며, 이 버전에서 자동으로 지원한다고 주장하지 않는다.
