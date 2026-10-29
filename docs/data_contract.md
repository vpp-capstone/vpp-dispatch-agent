# 데이터 전처리

## 1. 재생에너지 발전량 (Target)

전력거래소와 제주에너지공사 데이터를 비교해 메인 소스를 고르거나 서로 빈 구간을 메운다.

소스

- 한국전력거래소\_지역별 시간별 태양광 및 풍력 발전량
- 한국전력거래소\_지역별 시간별 태양광 발전량 정보
- 제주에너지공사\_풍력발전기 전력 정보
- 제주에너지공사\_태양광발전기 전력 정보

## 2. 전력 시장 및 수요

과거 SMP와 하루 전 예측 SMP의 차이, 전체 수요 대비 공급 여력(출력제어 시점)을 정리한다.

소스

| 자료 | 받는 곳 | 원본 넣는 곳 (`data/raw/market/`) | 모듈 (`vpp.market.`) |
| --- | --- | --- | --- |
| 한국전력거래소\_시간별 제주전력수요 | [포털 파일](https://www.data.go.kr/data/15065239/fileData.do), 과거 버전까지 전부 | `demand/*.csv` | `demand` |
| 한국전력거래소\_계통한계가격 및 수요예측(하루전 발전계획용) | [OpenAPI](https://www.data.go.kr/data/15131225/openapi.do) | `day_ahead_api/` (자동 저장) | `day_ahead` |
| 제주 실시간 SMP (엑셀) | [KPX 제주시범사업 정보 > SMP > 실시간](https://www.kpx.or.kr/menu.es?mid=a10406020200), 연도별 엑셀 | `realtime_xlsx/*.xlsx` | `realtime_xlsx` |
| 한국전력거래소\_제주시범사업정보 | [OpenAPI](https://www.data.go.kr/data/15133951/openapi.do) | `realtime_api/` (자동 저장) | `realtime_api` |
| 한국전력거래소\_전국 태양광 풍력 제어횟수 및 제주 풍력 제어량 | [포털 zip](https://www.data.go.kr/data/15132422/fileData.do), 제주 풍력·제주 태양광 CSV만 | `curtailment/*.csv` | `curtailment` |

- 제주 SMP는 하루전 가격을 `day_ahead`에서, 실시간(15분) 가격을 `realtime_api`와 `realtime_xlsx`에서 얻는다.
- 한국전력거래소\_제주 전력수급현황은 제외했다. 하루 한 줄(최대전력 시각 기준)이고 한 달 단위로만 받을 수 있으며, 최대전력은 시간별 수요에서 계산할 수 있다.
- OpenAPI 인증키는 `.env`의 `DATA_GO_KR_API_KEY`. 개발계정은 하루 100회 한도이고, 두 API 모두 전체 이력을 십여 번의 호출로 받는다.

실행 (앞의 다섯 개는 순서 무관, `build`는 마지막)

```bash
python -m vpp.market.demand
python -m vpp.market.day_ahead        # --build-only 를 붙이면 API를 다시 부르지 않음
python -m vpp.market.realtime_xlsx
python -m vpp.market.realtime_api     # --build-only 가능
python -m vpp.market.curtailment
python -m vpp.market.build
```

결과 (`data/processed/market/`). `ts`는 모두 KST 기준 **구간 시작 시각**이다 (원본 1시 = 00:00~01:00 → `00:00`).

| 파일 | 단위 | 기간 | 컬럼 |
| --- | --- | --- | --- |
| `market_hourly.csv` | 1시간 | 2020-01 ~ | 아래 표 |
| `jeju_rt_15min.csv` | 15분 | 2024-03 ~ | `smp_rt`, `fcst_rt`, `source`(api 또는 xlsx) |
| `jeju_demand_hourly.csv` | 1시간 | 2007-01 ~ | `demand_mwh`, `source`(원본 파일명) |
| `smp_forecast_hourly.csv` | 1시간 | 2020-01 ~ | `smp_land`, `smp_jeju`, `fcst_jeju`, `fcst_land`, `fcst_total` |
| `jeju_curtail_hourly.csv` | 1시간 | 2021-01 ~ 2024-06 | `wind_curtail_mwh`, `solar_curtail` |

`market_hourly.csv` 컬럼. 자료가 없는 구간은 빈 값이고, 컬럼마다 값이 있는 기간이 다르다.

| 컬럼 | 뜻 | 값이 있는 기간 |
| --- | --- | --- |
| `demand_mwh` | 제주 수요 실적 (MWh) | ~ 포털 최신 파일까지 |
| `fcst_jeju` | 제주 하루전 수요예측 | 2020-01 ~ |
| `smp_da` | 제주 하루전 SMP (원/kWh) | 2020-01 ~ |
| `smp_rt_mean`, `smp_rt_min`, `smp_rt_max` | 제주 실시간 SMP의 그 시간 4개 구간 평균·최저·최고 | 2024-03 ~ |
| `fcst_rt_mean` | 제주 실시간 수요예측의 그 시간 평균 | 2024-07 ~ |
| `wind_curtail_mwh` | 제주 풍력 출력제어량 (MWh), 없던 시간은 0 | 2021-01 ~ 2024-05 |
| `solar_curtail` | 제주 태양광 출력제어 여부 (1/0), 제어량은 원본에 없음 | 2021-01 ~ 2024-06 |

정리하면서 정한 것

- 수요: 파일마다 단위(MWh/kWh)·날짜 컬럼명·쉼표 표기·인코딩이 달라 통일했다. 기간이 겹치면 나중에 올라온 파일 값을 쓴다 (급락 구간이 메워진 정정본).
- 하루전: `hour` 표기가 `1`과 `01`로 섞여 있고 일부 날짜는 두 표기로 중복돼 있어 하나만 남긴다 (값은 같음).
- 실시간 SMP: 엑셀에는 값이 통째로 없는 날이 있어 API 값을 기본으로 쓰고, API에 없는 구간만 엑셀로 채운다. 두 출처가 겹치는 구간 중 일부(5일)는 값이 다르다.
- 출력제어: 원본에 없는 날은 제어가 없었던 날로 보고 0으로 채운다. 기록이 2024년 6월 초에서 끝나 그 뒤는 빈 값이다.
- 0원과 음수 SMP는 실제 값이다.

## 3. 날씨·날짜 (Feature)

관측과 예보의 시간축을 맞추고 평일/휴일을 원핫인코딩해 하나의 데이터프레임으로 묶는다.
입찰 마감이 전날 오전 11시이므로 그 이전에 발표된 예보만 사용한다.

소스

- 기상청\_지상(종관, ASOS) 시간자료 — 고산·서귀포·성산·제주, 2021~2025
- 기상청 API허브\_재생에너지 기상정보 — 위·경도 기반, 2021.06~2025.06
- 기상청 API허브\_천리안2A 일사량 — 위·경도 기반, 2019.08~
- 기상청\_단기예보 — 기온·습도·하늘상태·강수확률·풍향·풍속
- 한국천문연구원\_특일정보
- 제주특별자치도\_풍력·태양광 발전소 현황 — 발전소 위·경도

## 통합 전달 형식

| 항목                  | 내용              |
| --------------------- | ----------------- |
| 파일 위치·형식        | `data/processed/` |
| `timestamp` 규칙      |                   |
| 최종 컬럼 목록        |                   |
| 데이터 버전·출처 기록 |                   |

## 코드 연결

- `src/vpp/data.py` — `prepare_hourly`(수집·정제·저장), `load_hourly`(로딩)
- `src/vpp/forecast.py` — `predict_demand`(위 데이터 → `ForecastResponse`)
- `src/vpp/schemas.py` — 예측 출력 타입. 입력 필드가 필요하면 추가
