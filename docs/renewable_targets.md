# 제주 재생에너지 Target 정제

## 목적과 범위

담당한 네 데이터 소스를 비교하고, 예측 모델에 전달할 관측값과 품질 정보를 만든다.
공통 `data.py`, `schemas.py`, `data_contract.md`의 인터페이스는 변경하지 않는다.
이 모듈은 독립적인 파일 전처리 도구이며 서비스 간 REST API 연결은 아직 포함하지 않는다.

**현재 주 Target 후보는 제주 시간별 태양광·풍력 전력거래량(MWh)이다.**
이는 제주 전체 발전량이나 출력제어가 없을 때의 잠재 발전량과 다르다.
발전량 예측 담당자와 거래량을 정답으로 삼아도 되는지 먼저 합의해야 한다.
기존 화면 샘플의 `solar_mw`, `wind_mw`에 자동으로 덮어쓰지 않는다.

## 소스 비교

| 소스 | 확보한 기간 | 시간 단위 | 대상 | 현재 용도 |
| --- | --- | --- | --- | --- |
| [KPX 지역별 시간별 태양광 및 풍력 발전량](https://www.data.go.kr/data/15065269/fileData.do) | 2025년 전체 | 시간 | 제주 전력시장 참여 자원 | 주 Target 후보 |
| [KPX 지역별 시간별 태양광 발전량 정보 API](https://www.data.go.kr/data/15103243/openapi.do) | 미수집 | 시간 | API 설명 및 응답 확인 필요 | 파일 자료와 중복 검증 예정 |
| [제주에너지공사 풍력발전기 전력 정보](https://www.data.go.kr/data/15028181/fileData.do) | 2026-04-01~06-30 | 일 | 공사 운영 풍력단지 | 별도 참고 자료 |
| [제주에너지공사 태양광발전기 전력 정보](https://www.data.go.kr/data/15038930/fileData.do) | 2026-04-01~06-30 | 일 | 공사 운영 태양광단지 | 별도 참고 자료 |

파일 이름의 기준일이 실제 수록 기간 전체를 뜻하지는 않는다. 표는 확보한 파일 내용 기준이다.
KPX 파일에는 기존 ESS 충·방전이 반영될 수 있으며, 직접거래·자가소비 등은 포함 범위가 다르다.
야간 태양광의 작은 양수 값을 임의로 0으로 만들지 않는다.
제주에너지공사 파일의 수치 단위는 추가 확인이 필요하여 원래 값으로 보존한다.

### 서로 빈칸을 메울 수 있는가?

현재 확보한 KPX와 공사 자료는 날짜가 겹치지 않고 대상 설비와 시간 단위도 다르다.
따라서 어느 쪽이 더 정확한지 오차를 직접 계산하거나 상대 자료로 결측을 채우지 않는다.
일별 값을 24등분해 시간별 관측값을 만드는 것도 하지 않는다.
향후 같은 설비·같은 시각·같은 단위의 자료를 확보한 경우에만 대체 가능성을 검토한다.

## 실행

README의 설치 과정을 마친 후, 팀 Drive에서 다음 원본을 `data/raw/`에 둔다.

- `kpx_regional_renewables_20251231.csv`
- `jejuenergy_daily_solar_20260630.csv`
- `jejuenergy_daily_wind_20260630.csv`

저장소 루트에서 실행한다. PowerShell·Ubuntu 모두 한 줄 명령을 사용할 수 있다.

```bash
python -m vpp.renewable_targets --kpx-file data/raw/kpx_regional_renewables_20251231.csv --jec-solar data/raw/jejuenergy_daily_solar_20260630.csv --jec-wind data/raw/jejuenergy_daily_wind_20260630.csv --start 2025-01-01 --end 2025-12-31 --output-dir data/processed/targets
```

`--start`, `--end`는 양 끝을 포함한다. 생략하면 KPX 원본의 최소·최대 날짜를 사용한다.
날짜 범위를 명시해야 앞뒤가 통째로 빠진 기간도 결측으로 드러난다.
원본은 수정하지 않으며 원본·정제 CSV는 Git 제외 대상이다.

## 처리 순서

1. UTF-8 또는 CP949 CSV를 읽고 열 이름의 공백을 정리한다.
2. KPX에서 제주·태양광·풍력만 선택하고 날짜·시간 키를 검사한다.
3. 수치를 변환하고 동일 키 중복을 확인한다. 서로 다른 값이 충돌하면 정답에서 제외한다.
4. 날짜 × 거래시간 1~24 × 발전원 달력을 만들어 빠진 시간을 표시한다.
5. 결측·수치 오류·음수·충돌을 품질 플래그로 기록한다. 보간하지 않는다.
6. 공사 자료는 일별 형태를 유지하고 결측일·중복일·합계 일치 여부를 점검한다.
7. 결과 CSV, 예외 목록, 원본 SHA-256 및 품질 JSON을 저장한다.

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `renewable_targets_hourly.csv` | 제주 날짜·거래시간·발전원별 관측값과 품질 플래그 |
| `target_issues.csv` | 사용 불가능한 시간별 Target 행 |
| `kpx_conflicting_rows.csv` | 서로 다른 값이 있는 중복 키의 원본 행 |
| `jec_solar_daily_reference.csv` | 태양광 단지별 값과 공사 전체 합계, 일별 보존 |
| `jec_wind_daily_reference.csv` | 풍력 단지별 값과 공사 전체 합계, 일별 보존 |
| `quality_report.json` | 기간, 결측, 중복, 소스 URL, 원본 체크섬, 선택 이유 |

공사 파일의 `is_reported_total=True`는 단지 합계 행이다. 개별 단지와 전체 합계를 함께 더하지 않는다.
합계 비교 허용 차이는 원본 수치 단위로 0.01이며, 실제 발전량의 정확성을 검증하는 지표는 아니다.

### 시간별 결과의 주요 열

| 열 | 의미 |
| --- | --- |
| `trade_date`, `hour_label` | 원본 거래일 및 거래시간 1~24 |
| `fuel` | `solar` 또는 `wind` |
| `traded_energy_mwh` | 해당 시간의 거래 전력량 MWh |
| `raw_value`, `source_file`, `source_id` | 원본 값과 출처 |
| `quality_flag` | 아래 품질 상태 |
| `target_available` | 정답으로 사용할 관측값인지 여부 |
| `is_imputed` | 현재 모두 False |
| `includes_existing_ess` | 소스 집계에 기존 ESS가 반영되는 자료임을 표시; 개별 행의 ESS 가동 여부는 아님 |
| `hour_interval_convention` | 현재 `pending_team_confirmation` |

품질 상태:

- `observed`: 수치가 있고 음수·중복 충돌이 없는 관측값. 실제 0도 포함한다.
- `missing_hour`: 해당 키가 원본에 없다.
- `missing_or_invalid_value`: 키는 있지만 값이 비었거나 유효한 수치가 아니다.
- `conflicting_duplicate`: 같은 날짜·시간·발전원에 서로 다른 값이 있다.
- `negative_review`: 음수 값으로, 원본 확인 전 정답에서 제외한다.

학습·평가는 `target_available=True`를 사용한다. 결측을 0으로 바꾸지 않는다.
`observed`는 형식상 사용 가능한 관측이라는 뜻이며 센서 정확도나 모집단 적합성을 보장하지 않는다.
시간별 MWh를 평균 MW로 바꾸려면 구간 길이로 나눠야 한다. 수치가 같더라도 두 단위를 혼용하지 않는다.

## 남은 확인과 연결

1. **Target 정의 합의:** 거래량 예측인지 실제 발전량 예측인지 결정한다. 이 자료만으로 잠재 발전량·출력제어 손실은 식별할 수 없다.
2. **시각 의미 확인:** 거래시간 1이 어떤 구간을 뜻하는지 제공기관 문서로 확인한 뒤 KST timestamp와 기상자료를 연결한다. 현재 임의의 시작·종료 시각을 부여하지 않는다.
3. **공사 단위 확인:** 일별 수치가 전력량인지 평균 출력인지 확인한다.
4. **KPX API 수집:** 해당 공공데이터포털 API 인증키와 실제 응답 스키마를 확보한다. 기상청 API 키와는 별개다.
5. **API 비교:** 실제 응답을 `trade_date,hour_label,region,traded_energy_mwh` 열의 CSV로 명시적으로 매핑한 뒤 `--kpx-api-csv 경로`를 추가한다. `fuel`은 생략 시 solar로 처리한다. API의 시각·단위·지역 범위를 먼저 확인한다.
6. API가 제공되면 겹치는 키의 차이를 별도 `kpx_api_comparison.csv`에 기록한다. 비교 시 원본 전체 범위를 사용하며 자동 대체하지 않는다.
7. 팀 전달 형식이 합의되면 공통 데이터 모듈 및 기상 피처 결합에 연결한다.

현재 실행 결과는 [2026-10-10 품질 보고서](renewable_target_quality_20261010.md)에 정리했다.
