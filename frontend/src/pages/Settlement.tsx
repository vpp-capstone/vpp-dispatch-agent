import type { Snapshot, Page } from "../types";
import { difference, money, number, time } from "../lib/format";
import { Stats, SectionTitle, TaskRow } from "../components/Shared";

export function Settlement({
  data,
  setHour,
  navigate,
}: {
  data: Snapshot;
  setHour: (h: number) => void;
  navigate: (p: Page) => void;
}) {
  const delta = difference(data.actualIncome, data.plannedIncome);
  const issues = [...data.hours]
    .filter((h) => (h.shortfall ?? 0) > 0.001)
    .sort((a, b) => b.shortfall! - a.shortfall!)
    .slice(0, 3);
  return (
    <>
      <div className="eyebrow">{data.date} · 실적·정산</div>
      <h1>
        {data.actualIncome === null
          ? "실행 수익을 계산할 실측 결과가 필요합니다"
          : `${data.incomeLabel}은 ${money(data.actualIncome)}입니다`}
      </h1>
      <p className="subtitle">
        같은 설비·날짜·결과 묶음의 구성 항목을 비교합니다.
      </p>
      <Stats
        items={[
          { label: "계획 기준", value: money(data.plannedIncome) },
          { label: "실행 기준", value: money(data.actualIncome) },
          {
            label: "실행 − 계획",
            value: delta === null ? "비교값 미저장" : money(delta),
          },
        ]}
      />
      <SectionTitle
        title="수입 구성과 차이"
        aside="단위: 원 · 표시 금액은 만 원/억 원 환산"
      />
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>정산 항목</th>
              <th>계획</th>
              <th>실행</th>
              <th>차이</th>
            </tr>
          </thead>
          <tbody>
            {data.ledger.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{money(row.planned)}</td>
                <td className={(row.actual ?? 0) < 0 ? "negative" : ""}>
                  {money(row.actual)}
                </td>
                <td>{money(difference(row.actual, row.planned))}</td>
              </tr>
            ))}
            <tr className="total">
              <td>{data.incomeLabel}</td>
              <td>{money(data.plannedIncome)}</td>
              <td>{money(data.actualIncome)}</td>
              <td>{money(delta)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="caption">{data.incomeNote}</p>
      <SectionTitle title="다음 검토에 반영할 시간" />
      {issues.length ? (
        <div className="task-list">
          {issues.map((r) => (
            <TaskRow
              key={r.hour}
              when={time(r.hour)}
              title={`공급 부족 ${number(r.shortfall, 2)} MWh`}
              description="편차와 계획 방전·SOC를 함께 확인합니다."
              action="계획 재확인"
              onClick={() => {
                setHour(r.hour);
                navigate("plan");
              }}
            />
          ))}
        </div>
      ) : (
        <p className="empty-inline">
          {data.executionAvailable
            ? "선택일에 저장된 공급 부족이 없습니다."
            : "실행 결과 연결 후 편차 시간을 확인할 수 있습니다."}
        </p>
      )}
      <SectionTitle
        title="선택일까지의 누적 성과"
        aside={`누적 실행 ${money(data.cumulativeIncome)}`}
      />
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>기간</th>
              <th>계획 합계</th>
              <th>실행 합계</th>
              <th>차이</th>
            </tr>
          </thead>
          <tbody>
            {data.trend.map((r) => (
              <tr key={r.month}>
                <td>
                  {r.month}
                  {r.month === data.date.slice(0, 7)
                    ? ` · ${data.date.slice(8)}일까지`
                    : ""}
                </td>
                <td>{money(r.planned)}</td>
                <td>{money(r.actual)}</td>
                <td>{money(difference(r.actual, r.planned))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="bottom-action">
        <p>편차가 반복되면 운영 자원과 사업성 가정을 별도로 점검합니다.</p>
        <button className="button" onClick={() => navigate("resources")}>
          설비·사업성 재점검 ↗
        </button>
      </div>
    </>
  );
}
