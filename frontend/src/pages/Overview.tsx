import type { Dispatch } from "react";
import { ArrowRight } from "lucide-react";
import type { Snapshot, Page } from "../types";
import type { Action, Workspace } from "../lib/workspace";
import { number, money, time } from "../lib/format";
import {
  Stats,
  SectionTitle,
  TaskRow,
  ReplayControls,
} from "../components/Shared";
import { Timeline } from "../components/Timeline";

export function Overview({
  data,
  state,
  dispatch,
  navigate,
}: {
  data: Snapshot;
  state: Workspace;
  dispatch: Dispatch<Action>;
  navigate: (p: Page) => void;
}) {
  const point = data.hours[state.hour];
  const worst = data.hours.reduce((max, h) =>
    (h.shortfall ?? 0) > (max.shortfall ?? 0) ? h : max,
  );
  const received = data.hours.slice(0, state.hour + 1);
  const income = received.every((h) => h.income !== null)
    ? received.reduce((sum, h) => sum + h.income!, 0)
    : null;
  const inspect = (page: Page, h = state.hour) => {
    dispatch({ type: "hour", hour: h });
    navigate(page);
  };
  return (
    <>
      <div className="eyebrow">운영 현황 · 저장 결과 재생</div>
      <h1>
        {point.shortfall !== null && point.shortfall > 0.001
          ? `${time(state.hour)} 공급계획 대비 ${number(point.shortfall, 2)} MWh 부족`
          : data.executionAvailable
            ? "계획과 실행의 차이를 확인합니다"
            : "운영계획을 확인하고 실행 결과를 기다립니다"}
      </h1>
      <p className="subtitle">
        {data.date} · {data.title} · 선택한 결과 묶음의 시간별 흐름
      </p>
      <Stats
        items={[
          {
            label: "선택 시간 공급계획",
            value: `${number(point.plannedSupply, 2)} MWh`,
          },
          {
            label: "실행 공급",
            value:
              point.actualSupply === null
                ? "실측 대기"
                : `${number(point.actualSupply, 2)} MWh`,
          },
          {
            label: point.soc === null ? "계획 SOC" : "실행 SOC",
            value: `${number(point.soc ?? point.plannedSoc)}%`,
          },
          {
            label: "재생 시점까지 수입",
            value: money(income),
            hint: `00:00–${time(state.hour + 1)}`,
          },
        ]}
      />
      <SectionTitle
        title="하루의 운전 흐름"
        aside="시간 막대와 목록에서 검토할 구간을 선택하세요"
      />
      <Timeline
        hours={data.hours}
        selected={state.hour}
        onSelect={(h) => dispatch({ type: "hour", hour: h })}
      />
      <ReplayControls state={state} dispatch={dispatch} />
      <div className="focus-row">
        <div>
          <strong>
            {time(state.hour)} ·{" "}
            {point.charge !== null && point.charge > 0.001
              ? "ESS 충전 중"
              : point.discharge !== null && point.discharge > 0.001
                ? "ESS 방전 중"
                : data.executionAvailable
                  ? "ESS 대기"
                  : "확정 계획 검토"}
          </strong>
          <p>
            충전 {number(point.charge ?? point.plannedCharge, 2)} / 방전{" "}
            {number(point.discharge ?? point.plannedDischarge, 2)} MWh ·{" "}
            {point.soc === null ? "예측 기반 계획" : "실측 기반 실행 백테스트"}
          </p>
        </div>
        <button className="button primary" onClick={() => inspect("plan")}>
          해당 시간 계획 확인
          <ArrowRight size={15} />
        </button>
      </div>
      <SectionTitle title="이어서 할 일" aside="상황에서 필요한 업무로 이동" />
      <div className="task-list">
        <TaskRow
          when={time(worst.hour)}
          title={
            data.executionAvailable
              ? `최대 공급 편차 ${number(worst.shortfall, 2)} MWh 검토`
              : "선택한 날짜의 판매·충방전 계획 검토"
          }
          description="같은 날짜·설비·시간을 유지한 채 계획으로 이동합니다."
          action="계획 확인"
          onClick={() => inspect("plan", worst.hour)}
        />
        <TaskRow
          when={data.date.slice(5)}
          title={`${data.incomeLabel}과 정산 구성 확인`}
          description={
            data.actualIncome === null
              ? "실행값이 없는 항목은 미계산으로 구분합니다."
              : `하루 ${money(data.actualIncome)} · 같은 결과의 시간별 수입과 연결됩니다.`
          }
          action="실적 보기"
          onClick={() => inspect("settlement")}
        />
      </div>
    </>
  );
}
