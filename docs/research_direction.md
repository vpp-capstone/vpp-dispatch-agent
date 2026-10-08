# 연구 방향 (초안)

예측 정확도만 보는 게 아니라, **그 예측으로 ESS를 운영했을 때 비용이 얼마나 줄어드는지**까지 보려는 게 출발점.
아래 논문의 "예측 → 최적화 → 실제 비용" 구조를 참고해 단계별로 붙여 본다.

참고: Wanfu Zheng et al., _Smart predict-then-optimize-based model predictive control for
Virtual Power Plants with battery storage_, Journal of Energy Storage 142 (2026), 119606.
https://doi.org/10.1016/j.est.2025.119606

## 논문에서 가져올 아이디어

- 일반 방식: 예측 오차로 모델 학습 → 그 예측으로 최적화. 두 단계가 따로 논다.
- SPO 방식: 예측 `ŷ`로 계획 `a*`를 세우고, 실제 `y`에서의 비용 `J(a*, y)`를 예측 모델 학습에 되먹인다.
- 평가는 실측을 미리 아는 경우(Oracle)와의 비용 차이(regret)로 본다.

## 단계

1. 일반 예측 모델 — 예측 오차 기준으로 학습
2. 예측 → ESS 계획 → 실측 비용 평가 연결
3. 같은 기간에서 일반 모델 vs 비용 인지 모델 비교
4. 비용이 학습에 반영되도록 확장 (`forecast.py:train_cost_aware`)

예측 오차와 운영비용은 따로 보고한다. 오차가 낮다고 비용이 낮은 건 아님.

## 논문과 다른 점 (정해야함)

- 논문은 건물 부하·PV·배터리 + TOU 요금. 우리는 제주 SMP + 중개사업자 정산.
- 논문 타깃은 순부하(부하 − PV). 우리 타깃은 팀에서 결정.
- 포트폴리오 범위, ESS 제어 권한, 비용 항목, 출력제한, 종료 SOC 등은 `docs/formulation.md`에서 정리.
