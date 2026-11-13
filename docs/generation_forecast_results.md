# 기상예보 기반 발전량 예측 모델 결과

## 데이터와 평가 계약

- GFS 대상 기간: 2023-01-01~2025-12-31
- GFS 품질 통과: 1,095일 / 예상 1,096일 (99.91%)
- 누락일: 2023-05-24 한 건. 잘못된 파일은 없으며 해당 날짜는 학습마트에서 제외한다.
- 발전량 대상: 태양광 18개소, 풍력 3개 단지
- 정답 시각(`valid_time_kst`) 기준 2023~2024년: 개발 전용. 시간순 폴드와 2024년 전체
  홀드아웃으로 피처·파라미터·앙상블을 선택한 뒤 운영 모델을 재학습
- 2025년: 잠근 최종 테스트. 모델, 변수, 가중치 선택을 끝내고 운영 예측을 고정한 뒤 실측을
  결합해 한 번 평가했으며, 이 결과를 보고 모델을 다시 조정하지 않음
- 유효 정답: `label_is_valid == True`
- 발전량 지연값은 입찰 시점 누수 위험 때문에 사용하지 않음

기본 앙상블의 2024년 `validation_meta_fit` 지표는 결합 계수를 같은 기간에 학습한 적합도이므로
독립 성능으로 해석하지 않는다. 이후 개선 모델의 가중치는 2023년 3·4분기 OOF에서 정하고
2024년 전체를 홀드아웃으로 비교했다. 최종 일반화 성능은 동결된 모델로 2025년을 한 번만
평가해 확정한다.

## 모델

기준선은 2023년의 발전소·월·시간대별 평균 이용률이다. 모든 학습 모델은 이용률을 예측한 뒤
`[0, 1]`로 제한하고 설비용량을 곱해 kWh로 복원한다. 태양광은 태양고도가 0도 이하이거나
예보 일사량이 0이면 0kWh로 처리한다.

- 기본 LightGBM: 2023년 학습, 2024년 조기 종료, 확정 반복 수로 2023~2024년 재학습
- 기본 앙상블: LightGBM, HistGradientBoosting, ExtraTrees를 사용하고 2024년에서 Weighted와
  RidgeCV Stacking을 적합한 초기 벤치마크
- 최종 개선 모델: 2023년 OOF로 태양광 가중치와 풍력 동적 혼합 가중치를 정하고 2024년
  홀드아웃으로 구조를 선택한 뒤, 선택된 설정을 2023~2024년 전체에 재학습

## 2023~2024년 LightGBM 튜닝

2025년 발전량 실측은 ESS 수익 최종 시뮬레이션에 남겨두기 위해 튜닝에서 읽지 않는다.
튜닝 데이터 로더는 Parquet 필터로 `train`, `validation` 행만 읽고 연도가 2023~2024인지
다시 검사한다.

시간순 검증 구간은 2023년 3분기·4분기와 2024년 상반기·하반기다. 후보는 기존 설정,
설비용량·제곱근 설비용량 가중치, 잎 수와 규제 강도가 다른 LightGBM 설정 8개다. 다음 두
조건을 모두 만족하는 후보만 채택한다.

1. 네 시간 폴드의 `평균 nMAE + 0.25 × 표준편차`가 기존보다 낮음
2. 2023년 학습·2024년 전체 검증 nMAE도 기존보다 낮음

| 발전원 | 설정 | 시간 폴드 평균 nMAE | 안정성 점수 | 2024 전체 nMAE | 결정 |
|---|---|---:|---:|---:|---|
| 태양광 | 기존 | 3.8344% | 3.9650 | 3.7941% | 비교 기준 |
| 태양광 | 기존 파라미터 + `sqrt(capacity_kw)` 가중 | **3.8247%** | **3.9526** | **3.7829%** | 채택 |
| 풍력 | 기존 | **6.8650%** | **7.0813** | 6.4347% | 유지 |
| 풍력 | 15잎·강한 규제 + 용량 가중 | 6.8452% | 7.0570 | 6.5142% | 전체 검증 악화로 기각 |

태양광 개선 폭은 2024년 전체 기준 0.0112%p(상대 약 0.29%)로 작지만 시간 폴드 평균과
전체 검증이 모두 개선되어 반영했다. 풍력은 두 기준을 동시에 개선한 후보가 없어 기존 설정을
유지한다. 이후 AWS 피처 모델에서도 1일차·2일차 분리와 풍력 단지별 모델을 비교했다. 두
후보 모두 순차 폴드 평균은 개선했지만 2024년 전체 nMAE가 각각 6.2500%, 6.2443%로 공동
모델 6.2388%보다 높아 적용하지 않았다.

```bash
python -m vpp.generation_forecast.tune_lightgbm
```

튜닝 결과는 `data/processed/generation_forecast/generation_lightgbm_tuning_*.csv`와
`generation_lightgbm_tuning_summary.json`에 저장되며 Git에서는 제외된다.

## 풍력 파워커브·변화량 피처 실험

2025년 정답은 읽지 않고 동일한 4개 시간순 폴드와 2024년 전체 홀드아웃에서 다음 후보를
비교했다. 변화량은 같은 `asset_id + issue_time_utc` 예보 궤적 안에서만 계산하므로 다른
발행 시각이나 실측 미래값을 사용하지 않는다.

- 변화량 LightGBM: 1시간·3시간 풍속 차이와 원형 풍향 변화의 sin/cos 피처 추가
- 파워커브 + 잔차 LightGBM: 발전소별 0.5m/s 구간 단조 파워커브를 적합하고 남은 잔차를
  변화량 피처가 포함된 LightGBM으로 보정

| 풍력 후보 | 시간 폴드 평균 nMAE | 안정성 점수 | 최악 폴드 nMAE | 2024 전체 nMAE | 결정 |
|---|---:|---:|---:|---:|---|
| 기존 LightGBM | 6.8650% | 7.0813 | 7.8739% | 6.4347% | 비교 기준 |
| 풍향·풍속 변화량 LightGBM | 6.8339% | 7.0439 | 7.7914% | **6.3826%** | 발전소 모델에 반영 |
| 파워커브 + 잔차 LightGBM | 6.9419% | 7.1433 | 7.7600% | 6.4249% | 단독 미채택 |
| 방향별 파워커브 + 잔차 LightGBM | **6.8172%** | **6.9807** | **7.6900%** | 6.4438% | 중간 앙상블 후보 |

파워커브 잔차 모델은 2024년 전체 오차만 기존보다 소폭 개선됐고 시간 폴드 안정성은
악화되어 채택하지 않았다. 방향별 파워커브 모델도 2024년 단독 성능은 낮았지만 직접
포트폴리오 모델과의 상호 보완 가능성을 확인하기 위해 중간 OOF 앙상블 후보에 포함했다.
최종 선택에서는 궤적·공간 LightGBM 단일 모델보다 낮지 않아 운영 번들에서 제외했다. 실험은
다음 명령으로 재현하며 결과 CSV는
`data/processed/generation_forecast/wind_hybrid_*.csv`에 저장된다.

```bash
python -m vpp.generation_forecast.tune_wind_hybrid
```

출력제어량을 사용하지 않는 추가 실험에서는 같은 예보 발행분 안의 6시간 변화량, 앞쪽 예보
변화량, 인접 3시간 풍속 평균·범위, U/V 성분 변화량, 풍속 제곱·세제곱과 풍력 단지가 연결된
GFS 2개 격자의 평균·편차·범위를 추가했다.

| 풍력 후보 | 시간 폴드 평균 nMAE | 안정성 점수 | 최악 폴드 nMAE | 2024 전체 nMAE |
|---|---:|---:|---:|---:|
| 기존 변화량 LightGBM | 6.8495% | 7.0669 | 7.8743% | 6.3665% |
| **궤적·공간 피처 LightGBM** | 6.7133% | 6.9079 | 7.5403% | **6.2517%** |
| 고급 피처 직접 포트폴리오 | **6.6615%** | **6.8271** | **7.3158%** | 6.3879% |

궤적·공간 피처 LightGBM은 2024년 상·하반기와 1일차·2일차가 모두 개선됐다. 2023년 OOF
가중 앙상블은 6.2763%로 단일 모델보다 낮지 않아, 최종 풍력 운영 모델은 이 LightGBM
하나로 단순화했다.

```bash
python -m vpp.generation_forecast.tune_wind_improvements
```

## 발전량 정답 진단과 최종 개선 모델

정답 진단은 한 실제 시각에 연결된 여러 예보 행을 한 번만 세어 수행했다. 2023~2024년에서
음수 정답은 없었고, 풍력 결측은 한경 45시간·성산 4시간이었다. 고일사·저출력 및
고풍속·저출력 구간은 정비·출력제한일 가능성이 있지만 기상예보 오차와 구분할 근거가 없어
자동 삭제하지 않고 진단 플래그만 남겼다.

태양광은 clear-sky 지수, 일사·운량 변화량, 제주 격자 간 평균·편차·범위를 만들었다. 이 피처를
발전소별 모델에 바로 추가한 후보는 2024년 전체에서 개선되지 않았지만, 전체 포트폴리오를
직접 예측하는 모델에서는 효과가 있었다. 최종 가중치는 2024년이 아니라 2023년 3·4분기
시간순 OOF 예측만으로 학습했다.

| 발전원 | 비교 모델 | 2024 전체 | 상반기 | 하반기 | 1일차 | 2일차 |
|---|---|---:|---:|---:|---:|---:|
| 태양광 | 발전소 LightGBM 합산 | 3.7842% | 4.1400% | 3.4324% | 3.6550% | 3.9135% |
| 태양광 | **OOF 가중 앙상블** | **3.6623%** | **3.9807%** | **3.3473%** | **3.5298%** | **3.7948%** |
| 풍력 | 변화량 LightGBM 합산 | 6.3826% | 6.8296% | 5.9405% | 6.0437% | 6.7217% |
| 풍력 | OOF 가중 앙상블 | 6.2763% | 6.6945% | 5.8627% | 5.9377% | 6.6150% |
| 풍력 | **궤적·공간 LightGBM** | **6.2517%** | **6.7069%** | **5.8013%** | **5.9062%** | **6.5971%** |

- 태양광 가중치: 발전소 LightGBM 0.3503, 직접 포트폴리오 0.6497
- 풍력 OOF 가중치는 비교 후보로만 보존하고, 운영 모델은 궤적·공간 LightGBM 1.0 사용
- 시간대·예보일별 OOF 편향 보정은 태양광 3.7515%, 풍력 6.5451%로 악화되어 기각
- 태양광 직접 포트폴리오 예측에도 야간 0 규칙을 적용

```bash
python -m vpp.generation_forecast.audit_labels
python -m vpp.generation_forecast.tune_solar_improvements
python -m vpp.generation_forecast.tune_wind_improvements
python -m vpp.generation_forecast.tune_oof_ensemble
python -m vpp.generation_forecast.improved_model
```

## AWS 기반 풍력 보정 실험

기상청 AWS 2023~2024년 시간자료는 발전시각의 실측값을 모델 입력에 직접 넣지 않았다.
각 검증 폴드의 학습기간에 해당하는 관측만 사용해 `GFS 80m 예보 -> 지역 AWS 바람 벡터`
Ridge 보정기를 학습하고, 검증기간에는 GFS 예보만으로 AWS 프록시 피처를 생성했다.

- 한경풍력: 낙천(990)
- 상명풍력: 새별오름(883). 금악(880)은 2023년 파일이 없어 자동 제외
- 성산풍력: 성산수산(892)
- 여러 주변 지점을 평균한 후보보다 각 단지의 주 지점만 사용한 후보가 우수
- 2024 전체 풍력 포트폴리오 nMAE: `6.2517% -> 6.2388%`
- 2025 정답은 로드하거나 재평가하지 않음

```bash
python -m vpp.generation_forecast.tune_aws_wind
python -m vpp.generation_forecast.improved_model
```

AWS 원본은 `data/raw/aws/`에 두며 Git에는 포함하지 않는다. 운영 번들의
`wind_improved_production.joblib`에는 학습된 보정기가 들어 있어 추론 시 AWS 실측 파일을
다시 읽지 않고 GFS 예보만으로 동일한 프록시 피처를 계산한다.

## AWS 기반 태양광 보정 실험

AWS 일사·일조 열은 구좌(781)의 2023년 일부 행을 제외하면 비어 있었고, 구좌 값도 모두
0이어서 피처로 사용하지 않았다. 대신 거의 완전하게 제공된 기온·습도를 사용했다. 각
검증 폴드의 과거 구간에서만 `GFS 일사·운량·기온 -> 인근 AWS 기온·습도` Ridge 보정기를
학습하고, 검증기간에는 GFS 예보만으로 다음 프록시를 생성했다.

- AWS 보정 기온·습도
- GFS 대비 기온 편차
- 이슬점과 수증기압 부족량(VPD)
- 습도×운량, 기온×일사 상호작용

AWS 피처를 모든 태양광 구성요소에 넣으면 순차 검증이 악화됐다. 최종적으로 AWS 피처는
발전소별 LightGBM에만 적용하고 기존 직접 포트폴리오 모델은 유지했다.

- 기존 OOF 가중 결합 2024 nMAE: `3.6623%`
- AWS 자산 모델 + 기존 직접 모델: `3.6564%`
- 순차 검증 선택점수: `3.8865 -> 3.8829`
- 최악 폴드 nMAE: `4.4727% -> 4.4538%`
- 2025 정답은 로드하거나 재평가하지 않음

```bash
python -m vpp.generation_forecast.tune_aws_solar
python -m vpp.generation_forecast.improved_model
```

최종 `solar_improved_production.joblib`에는 18개 발전소별 AWS 보정기가 포함된다. 추론 시
AWS 파일이나 미래 관측을 읽지 않고 GFS 예보만으로 프록시를 계산한다.

마지막 명령은 2025년을 읽지 않고 2023~2024년 전체로 운영 번들을 재학습한다.

## AWS 풍력 LightGBM 추가 튜닝

AWS 보정 피처를 적용한 뒤 LightGBM 구조를 2023~2024년 시간순 4개 폴드에서 다시
비교했다. 낮은 학습률과 강화된 최소 리프 표본을 사용한 `slow_31` 후보가 기존 설정보다
롤링 안정성과 2024년 전체 성능을 함께 개선해 최종 풍력 번들에 반영했다.

- 파라미터: `learning_rate=0.015`, `num_leaves=31`, `min_child_samples=150`,
  `reg_lambda=1.0`
- 롤링 선택 점수: `6.8809 -> 6.8328`
- 2024년 포트폴리오 nMAE: `6.2388% -> 6.2177%`
- 2025년 정답 로드 여부: `false`

```bash
python -m vpp.generation_forecast.tune_aws_wind_lgbm
python -m vpp.generation_forecast.improved_model
```

## 풍향 섹터·풍속 구간 동적 혼합과 불확실성

AWS 프록시 풍향을 8개 섹터로 나눈 범주형 피처는 롤링 선택점수와 2024년 전체 nMAE를 모두
소폭 개선했다. 여기에 AWS 프록시 풍속을 `<3m/s`, `3~7m/s`, `>=7m/s`로 나눠 학습한
전문가 LightGBM과 전체 구간 공통 LightGBM을 결합했다. 결합 가중치는 2023년 3·4분기 OOF
예측에서만 풍속 구간별로 정했다.

| 풍력 후보 | 롤링 선택점수 | 2024 전체 nMAE | 결정 |
|---|---:|---:|---|
| AWS `slow_31` | 6.8328 | 6.2177% | 비교 기준 |
| 풍향 8섹터 `slow_31` | 6.8301 | 6.2129% | 공통 모델에 반영 |
| 풍속 3구간 전문가 단독 | 6.9214 | 6.1954% | 롤링 안정성 악화로 단독 기각 |
| **공통+전문가 동적 혼합** | **6.7669** | **6.1712%** | 채택 |

- 구간별 전문가 가중치: 저풍속 0.7610, 중풍속 0.0000, 고풍속 0.7133
- CatBoost OOF 결합은 롤링 점수는 개선했지만 2024년 전체가 6.2437%로 악화되어 기각
- 교차검증 잔차 LightGBM/Ridge는 최선이 6.8617%로 악화되어 운영 모델에 넣지 않음
- clear-sky 정규화 태양광 목표도 2024년 3.6787%로 기존 3.6564%보다 낮지 않아 기각

마지막 개선 단계에서는 단지별 모델, 비선형 AWS 보정, 제한적 편향 보정, 시간 연속성
후처리를 순서대로 비교했다.

| 단계 | 롤링 선택점수 | 2024 전체 nMAE | 결정 |
|---|---:|---:|---|
| 공통+전문가 동적 혼합 | 6.7669 | 6.1712% | 비교 기준 |
| **단지별 모델 혼합** | **6.6617** | **6.1426%** | 채택 |
| 비선형 AWS LightGBM 보정 | 6.9542 | 6.3258% | 기각 |
| 제한적 그룹 중앙값 편향 보정 | 6.7078 | 6.1177% | 롤링 악화로 기각 |
| **3시간 이동평균 25% 혼합** | **6.6418** | **6.1255%** | 채택 |

단지별 모델 혼합 비율은 한경 0.3641, 상명 1.0, 성산 0.6442다. 나머지 비중은 공통·전문가
동적 혼합 모델을 사용한다. 평활은 같은 `asset_id + issue_time_utc`의 예보 궤적 안에서만
수행하며, 중심 3시간 이동평균을 원래 예측에 25%만 섞는다. 실제 미래 발전량이나 다른 예보
발행분은 사용하지 않는다.

풍력 P10/P50/P90은 2023년 OOF 잔차의 예보일·풍속 구간별 가중 분위수로 보정한다. P50은
점 예측을 그대로 유지한다. 2023년만으로 구간을 보정하고 2024년에서 확인한 P10~P90의
경험적 포함률은 82.00%(목표 80%), 평균 구간 폭은 이용률 0.2127이었다. 운영 번들에는 이후
확보된 2024년 OOF 잔차까지 포함해 2025년 추론용 보정기를 다시 적합하되 2025년 정답은
읽지 않는다.

```bash
python -m vpp.generation_forecast.tune_wind_direction_interactions
python -m vpp.generation_forecast.tune_wind_dynamic_regime_ensemble
python -m vpp.generation_forecast.tune_wind_asset_models
python -m vpp.generation_forecast.tune_wind_postprocessing
python -m vpp.generation_forecast.evaluate_wind_uncertainty
python -m vpp.generation_forecast.improved_model
```

## 현재 최종 선택 모델

2025년을 열기 전 동결한 현재 운영 번들은 다음과 같다. 아래 nMAE는 모두 개발 기간인 2024년
전체 홀드아웃 수치이며 최종 일반화 성능으로 해석하지 않는다.

| 발전원 | 현재 구성 | 2024 전체 nMAE | 운영 번들 |
|---|---|---:|---|
| 태양광 | AWS 보정 발전소 LightGBM 0.3503 + 직접 포트폴리오 LightGBM 0.6497 | 3.6564% | `solar_improved_production.joblib` |
| 풍력 | 공통·풍속전문가 동적 혼합 + 단지별 모델 혼합 + 3시간 약한 평활 | 6.1255% | `wind_improved_production.joblib` |

두 번들 모두 2023~2024년 전체로 재학습했고 AWS Ridge 보정기를 내부에 포함한다. 추론에는
GFS 예보만 필요하며 2025년 AWS 실측은 필요하지 않다. 풍력 번들은 발전량 점 예측과 함께
P10/P50/P90을 출력한다. `generation_improved_production_summary.json`의
`locked_2025_loaded`는 `false`다.

운영 예측을 고정한 뒤 같은 예측값에 2025년 실측을 결합해 최종 평가했다. 포트폴리오 48시간
전체 nMAE는 태양광 `3.941932%`, 풍력 `5.670580%`다. 태양광은 완전한 포트폴리오 시각
17,520개를 평가했고, 풍력은 실측 결측 26행을 제외해 17,494개를 평가했다. 이 결과를 보고
모델·피처·가중치를 다시 선택하지 않는다.

아래 수치는 현재 최종 모델을 만들기 전 기본 모델·기본 앙상블의 참고 벤치마크다.

## 기존 기본 모델의 2025년 튜닝 전 잠금 테스트

포트폴리오 48시간 전체의 nMAE다. 낮을수록 좋다. 아래 값은 현재 최종 모델의 성능이 아니라
이번 2023~2024 개선 전에 기본 모델·기본 앙상블로 생성한 참고 벤치마크다.

| 발전원 | 기준선 | LightGBM | HistGB | ExtraTrees | Weighted | Stacking |
|---|---:|---:|---:|---:|---:|---:|
| 태양광 | 6.0643% | 4.2839% | 4.4014% | **4.0824%** | 4.1358% | 4.1224% |
| 풍력 | 13.2916% | **6.1357%** | 6.7320% | 6.8572% | **6.1357%** | 6.2156% |

| 발전원·모델 | 1일차 nMAE | 2일차 nMAE |
|---|---:|---:|
| 태양광 LightGBM | 4.2044% | 4.3633% |
| 태양광 ExtraTrees | **3.9863%** | **4.1785%** |
| 태양광 Weighted | 4.0528% | 4.2187% |
| 태양광 Stacking | 4.0239% | 4.2208% |
| 풍력 LightGBM/Weighted | **5.8636%** | **6.4078%** |

- 태양광은 LightGBM 단독보다 ExtraTrees, Weighted, Stacking이 모두 개선됐다. nMAE 기준 단일
  ExtraTrees가 가장 낮고, RMSE는 Stacking이 가장 낮다.
- 풍력 Weighted 가중치는 LightGBM 1.0으로 수렴했다. 따라서 현재 데이터에서는 복잡한 앙상블을
  운영할 이득이 없으며 LightGBM 단독이 가장 단순한 선택이다.
- 2일차 오차가 1일차보다 크므로 ESS 최적화에서 예측거리별 불확실성을 따로 고려하는 편이 좋다.

## 기존 기본 앙상블 학습 결과

| 발전원 | LightGBM | HistGB | ExtraTrees | Stacking alpha |
|---|---:|---:|---:|---:|
| 태양광 | 0.5758 | 0.0000 | 0.4242 | 100 |
| 풍력 | 1.0000 | 0.0000 | 0.0000 | 100 |

이 표는 초기 기본 앙상블의 결과이며 현재 태양광 OOF 가중치와는 다르다. 해당 가중치와
Stacking 계수는 2024년에서 고정한 뒤 기존 2025년 벤치마크에 적용했다. 2025년 결과를 보고
다시 튜닝하면 최종 테스트의 의미가 사라지므로 추가 변경은 새로운 검증 기간을 확보한 뒤 한다.

## 산출물

아래 파일은 모두 Git 제외 대상이다. 팀 Drive로 공유한 뒤 같은 경로에 배치한다.

- `artifacts/models/`
  - `*_lightgbm_selection.joblib`: 2023년 학습·2024년 평가용 LightGBM
  - `*_lightgbm.joblib`: 2023~2024년 재학습 운영용 LightGBM
  - `*_ensemble.joblib`: 2023년 기본 모델, 가중치, Stacking 메타 모델 묶음
  - `*_improved_production.joblib`: 2023~2024년 재학습 최종 선택 모델. 태양광은 OOF 가중
    앙상블, 풍력은 풍향 섹터 공통 모델·풍속 구간 전문가·단지별 모델·평활·불확실성 보정기 묶음
- `artifacts/predictions/`
  - `*_validation_predictions.parquet`, `*_test_predictions.parquet`: LightGBM 예측
  - `*_ensemble_validation_meta_fit_predictions.parquet`: 2024년 앙상블 적합 예측
  - `*_ensemble_test_predictions.parquet`: 개선 전 기본 앙상블의 2025년 참고 예측. 최신
    `*_improved_production.joblib`의 최종 예측이 아님
- `artifacts/2024_validation/`
  - `*_improved_2024_validation_with_actual.parquet`: 2023년 라벨로 재학습한 개발 검증 파일.
    일부 계획일은 유효 실측 행이 빠져 48시간 창이 불완전하므로 운영 파이프라인에서 제외됨
- `artifacts/2025/operational/`
  - `*_improved_predictions.parquet`: 실측 열이 없는 고정 운영 예측
- `artifacts/2025/evaluation/`
  - `*_improved_2025_with_actual.parquet`: 고정 운영 예측에 2025년 실측만 결합한 최종 평가 파일
- `artifacts/generation_evaluation_export_summary.json`
  - 평가 파일별 행 수와 SHA-256. 기간과 nMAE는 이 문서에 기록
- `data/processed/generation_forecast/`
  - `generation_model_metrics.csv`, `generation_asset_metrics.csv`, `training_summary.json`
  - `generation_ensemble_metrics.csv`, `generation_ensemble_asset_metrics.csv`
  - `generation_ensemble_summary.json`
  - `generation_label_audit.*`, `*_improvement_*.csv`
  - `generation_oof_ensemble_2024.csv`, `generation_oof_ensemble_summary.json`
  - `generation_improved_production_summary.json`
  - `wind_aws_lgbm_tuning_*.csv`, `wind_aws_lgbm_tuning_summary.json`
  - `wind_dynamic_regime_*`, `wind_asset_model_*`, `wind_postprocessing_*`,
    `wind_uncertainty_*`

예측 파일의 고유키는 `asset_id + issue_time_utc + valid_time_utc`다. 포트폴리오 예측은 같은
`issue_time_utc + valid_time_utc`의 발전소별 예측 kWh를 합산한다.

## 재학습과 최종 번들 생성

학습마트와 `data/raw/aws/`가 준비된 저장소 루트에서 실행한다. 초기 기본 모델과 기본
앙상블 벤치마크를 다시 만들려면 다음 명령을 사용한다.

```bash
python -m vpp.generation_forecast.model
python -m vpp.generation_forecast.ensemble
```

최종 개선 번들을 깨끗한 결과 디렉터리에서 재현하려면 선택 보고서를 먼저 생성한 뒤 마지막에
운영 모델을 학습한다.

```bash
python -m vpp.generation_forecast.tune_oof_ensemble
python -m vpp.generation_forecast.tune_aws_wind
python -m vpp.generation_forecast.tune_aws_solar
python -m vpp.generation_forecast.tune_aws_wind_lgbm
python -m vpp.generation_forecast.tune_wind_direction_interactions
python -m vpp.generation_forecast.tune_wind_dynamic_regime_ensemble
python -m vpp.generation_forecast.tune_wind_asset_models
python -m vpp.generation_forecast.tune_wind_postprocessing
python -m vpp.generation_forecast.evaluate_wind_uncertainty
python -m vpp.generation_forecast.improved_model
```

마지막 명령은 `generation_oof_ensemble_summary.json`, `wind_aws_2024_holdout.csv`,
`solar_aws_2024_holdout.csv`, `wind_aws_lgbm_tuning_summary.json`,
`wind_dynamic_regime_summary.json`, `wind_asset_model_summary.json`,
`wind_postprocessing_summary.json`, `wind_final_fold_predictions.parquet`을 선택 계약으로 읽는다.
운영 번들은 2023~2024년만 재학습하며 2025년 정답을 읽지 않는다. 재실행하면 같은 경로의
Git 제외 모델·지표 파일을 갱신한다.
