import { useState } from "react";
import type { Snapshot, Page } from "../types";
import { money, number, time } from "../lib/format";
import { SectionTitle, Stats } from "../components/Shared";
import { Timeline } from "../components/Timeline";
import { Term } from "../components/Help";
import { useReview } from "../hooks/useReview";

export function Plan({
  data,
  hour,
  setHour,
  navigate,
}: {
  data: Snapshot;
  hour: number;
  setHour: (h: number) => void;
  navigate: (p: Page) => void;
}) {
  const review = useReview(data.id);
  const [memo, setMemo] = useState(review.value.memo);
  const completed = review.value.complete && memo === review.value.memo;
  const sum = (key: "plannedCharge" | "plannedDischarge" | "plannedSupply") =>
    data.hours.reduce((a, h) => a + h[key], 0);
  return (
    <>
      <div className="breadcrumb">
        <b>운영계획 검토</b>
        <span>→</span>
        <button onClick={() => navigate("overview")}>운영 현황</button>
        <span>→</span>
        <button onClick={() => navigate("settlement")}>실적·정산</button>
      </div>
      <div className="eyebrow">{data.date} · 저장된 확정 계획</div>
      <h1>판매·충방전 계획과 잔량을 검토합니다</h1>
      <p className="subtitle">
        현재 결과 묶음의 계획을 조회합니다. 조건 변경·재계산은 기존 Python
        파이프라인에서 수행합니다.
      </p>
      <Stats
        items={[
          {
            label: "하루 공급계획",
            value: `${number(sum("plannedSupply"))} MWh`,
          },
          {
            label: "ESS 충전 / 방전",
            value: `${number(sum("plannedCharge"))} / ${number(sum("plannedDischarge"))} MWh`,
          },
          {
            label: "계획 기준 수입",
            value: money(data.plannedIncome),
            hint:
              data.plannedIncome === null
                ? "이 배분 기준의 계획 수입 미저장"
                : data.incomeLabel,
          },
        ]}
      />
      <SectionTitle
        title="시간대별 공급계획"
        aside={`검토 시간 ${time(hour)}`}
      />
      <Timeline
        hours={data.hours}
        selected={hour}
        onSelect={setHour}
        planOnly
      />
      <div className="time-control">
        <label htmlFor="plan-hour">확인할 시간</label>
        <input
          id="plan-hour"
          type="range"
          min="0"
          max="23"
          value={hour}
          onChange={(e) => setHour(Number(e.target.value))}
        />
        <output>{time(hour)}</output>
      </div>
      <SectionTitle
        title="검토 구간의 운전 조치"
        aside="인접 5시간 · 전체 표는 아래에서 확인"
      />
      <PlanTable data={data} hour={hour} compact />
      <details className="details">
        <summary>24시간 확정 계획 전체</summary>
        <PlanTable data={data} hour={hour} />
      </details>
      <div className="review-form">
        <label htmlFor="plan-memo">검토 메모</label>
        <textarea
          id="plan-memo"
          maxLength={500}
          value={memo}
          onChange={(e) => setMemo(e.target.value)}
          placeholder="실행 시 확인할 편차와 잔량 조건을 기록하세요."
        />
        <div className="bottom-action">
          <p>
            검토 기록은 이 브라우저에 저장됩니다. 실제 입찰·설비 명령은 전송하지
            않습니다.
          </p>
          <button
            className="button primary"
            onClick={() =>
              completed
                ? navigate("overview")
                : review.save({ memo, complete: true })
            }
          >
            {completed ? "운영 현황에서 결과 재생" : "검토 완료 기록"}
          </button>
        </div>
        <p className="status" role="status">
          {review.error ||
            (completed ? "이 결과 묶음에 대한 검토 완료 기록이 있습니다." : "")}
        </p>
      </div>
    </>
  );
}
function PlanTable({
  data,
  hour,
  compact = false,
}: {
  data: Snapshot;
  hour: number;
  compact?: boolean;
}) {
  const rows = compact
    ? data.hours.slice(
        Math.min(19, Math.max(0, hour - 2)),
        Math.min(24, Math.max(5, hour + 3)),
      )
    : data.hours;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>시간</th>
            <th>
              <Term label="공급계획" />
            </th>
            <th>
              <Term label="ESS 충전" />
            </th>
            <th>
              <Term label="ESS 방전" />
            </th>
            <th>
              <Term label="종료 SOC" />
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.hour} className={hour === r.hour ? "selected" : ""}>
              <td>{time(r.hour)}</td>
              <td>{number(r.plannedSupply, 2)} MWh</td>
              <td>{number(r.plannedCharge, 2)} MWh</td>
              <td>{number(r.plannedDischarge, 2)} MWh</td>
              <td>{number(r.plannedSoc)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
