import { useReducer, useState } from "react";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import type { Catalog, Candidate } from "../types";
import { lineLabel, money, number } from "../lib/format";
import { SectionTitle } from "../components/Shared";
import { comparisonReducer } from "../lib/comparison";
import {
  candidateView,
  demoComparison,
  type CandidateScope,
} from "../lib/resourceCandidates";
import { useReview } from "../hooks/useReview";

export function Resources({
  catalog,
  mode = "review",
  active,
  onSelect,
  onBack,
}: {
  catalog: Catalog;
  mode?: "onboarding" | "review";
  active: string;
  onSelect: (id: string) => void;
  onBack: () => void;
}) {
  const onboarding = mode === "onboarding";
  const demo = demoComparison(catalog.candidates);
  const current = catalog.candidates.find(
    (c) => c.id === active && c.hasArchive,
  );
  const [scope, setScope] = useState<CandidateScope>("ready");
  const [line, setLine] = useState(
    onboarding ? demo.line : (current?.line ?? demo.line),
  );
  const [comparison, dispatch] = useReducer(
    comparisonReducer,
    onboarding
      ? { ids: demo.ids, chosen: null }
      : {
          ids: current ? [current.id] : [],
          chosen: current?.id ?? null,
        },
  );
  const { lines, rows } = candidateView(catalog.candidates, scope, line);
  const readyCount = catalog.candidates.filter((c) => c.hasArchive).length;
  const allLineCount = catalog.candidates.filter((c) => c.line === line).length;
  const changeScope = (next: CandidateScope) => {
    const view = candidateView(catalog.candidates, next, line);
    setScope(next);
    setLine(view.line);
    dispatch({ type: "retain", ids: view.rows.map((c) => c.id) });
  };
  const resetDemo = () => {
    setScope("ready");
    setLine(demo.line);
    dispatch({ type: "preset", ids: demo.ids });
  };
  const compared = comparison.ids.flatMap((id) =>
    rows.filter((c) => c.id === id),
  );
  const candidate = compared.find((c) => c.id === comparison.chosen);
  const limitReached = compared.length === 2;
  return (
    <div className={onboarding ? "onboarding" : "resource-review"}>
      <div className="resource-back">
        <button className="text-button" onClick={onBack}>
          <ArrowLeft size={15} />
          {onboarding ? "처음으로" : "운영 업무로 돌아가기"}
        </button>
        <span>
          {onboarding
            ? "첫 이용 / 설비·사업성 검토"
            : "추가 사업 / 설비·사업성 재점검"}
        </span>
      </div>
      {onboarding && (
        <ol className="entry-progress" aria-label="시작 순서">
          <li aria-current="step">01 설비·사업성 검토</li>
          <li>02 선택안 확정</li>
          <li>03 운영계획</li>
        </ol>
      )}
      <div className="eyebrow">
        {onboarding
          ? "첫 단계 · 비교하고 결정하기"
          : "비정기 검토 · 별도 분석 공간"}
      </div>
      <h1>
        {onboarding
          ? "우리 사업에 맞는 설비를 검토합니다"
          : "설비와 사업성 조건을 다시 점검합니다"}
      </h1>
      <p className="subtitle">
        배전망 ESS · 태양광 5.7 MW의 저장된 후보를 비교합니다. 21곳 운영
        포트폴리오와 별도 분석 범위입니다.
      </p>
      <div className="candidate-scope">
        <div className="scope-buttons" role="group" aria-label="후보 표시 범위">
          <button
            className={scope === "ready" ? "active" : ""}
            aria-pressed={scope === "ready"}
            onClick={() => changeScope("ready")}
          >
            운영 시연 가능 <span>{readyCount}</span>
          </button>
          <button
            className={scope === "all" ? "active" : ""}
            aria-pressed={scope === "all"}
            onClick={() => changeScope("all")}
          >
            전체 후보 <span>{catalog.candidates.length}</span>
          </button>
        </div>
        <button
          className="text-button"
          disabled={!demo.ids.length}
          onClick={resetDemo}
        >
          기본 시연 비교
        </button>
      </div>
      <p className="caption">
        {scope === "ready"
          ? "저장 결과 검증을 통과해 운영계획·현황·정산까지 확인할 수 있는 후보입니다. 다른 설비는 ‘전체 후보’에서 계속 검토할 수 있습니다."
          : "모든 설비의 사업성을 비교합니다. 운영 시뮬레이션이 준비되지 않은 후보도 목록에 유지됩니다."}
      </p>
      <div className="resource-filter">
        <label htmlFor="line">선로 조건</label>
        <select
          id="line"
          value={line}
          disabled={!lines.length}
          onChange={(e) => {
            setLine(e.target.value);
            dispatch({ type: "reset" });
          }}
        >
          {!lines.length && <option value="">시연 가능한 선로 없음</option>}
          {lines.map((l) => (
            <option key={l} value={l}>
              {lineLabel(l)}
            </option>
          ))}
        </select>
        <span>기준 시나리오 · 표준 배분 · ESS 용량정산금 제외</span>
      </div>
      <SectionTitle
        title="동일 선로의 설비 후보"
        aside={`현재 선로 ${rows.length}개 표시 / 전체 ${allLineCount}개 · NPV는 보고값 기준`}
      />
      <p className="comparison-guide" id="comparison-limit" role="status">
        <strong>{compared.length} / 2개 선택</strong>
        {limitReached
          ? "아래에서 두 후보를 비교하세요. 다른 후보를 추가하려면 선택 하나를 해제하세요."
          : "왼쪽에서 최대 2개를 체크하면 아래에서 직접 비교할 수 있습니다."}
      </p>
      <div
        className="table-scroll candidate-table"
        tabIndex={0}
        aria-label="설비 후보 비교 목록"
      >
        <table>
          <colgroup>
            <col className="candidate-check-col" />
            <col className="candidate-name-col" />
            <col className="candidate-value-col" />
            <col className="candidate-value-col" />
            <col className="candidate-archive-col" />
          </colgroup>
          <thead>
            <tr>
              <th scope="col">선택</th>
              <th scope="col">설비</th>
              <th scope="col">연간 VPP 수입</th>
              <th scope="col">20년 NPV</th>
              <th scope="col">운영 시뮬레이션</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr
                key={c.id}
                className={comparison.ids.includes(c.id) ? "selected" : ""}
              >
                <td className="candidate-check">
                  <input
                    type="checkbox"
                    aria-label={`${c.label} 비교 선택`}
                    aria-describedby="comparison-limit"
                    checked={comparison.ids.includes(c.id)}
                    disabled={limitReached && !comparison.ids.includes(c.id)}
                    onChange={() => dispatch({ type: "toggle", id: c.id })}
                  />
                </td>
                <td className="candidate-name">
                  {c.label}
                  {!onboarding && c.id === active && (
                    <small className="inline-label">현재 조회</small>
                  )}
                </td>
                <td>{money(c.income)}</td>
                <td className={(c.npv ?? 0) < 0 ? "negative" : ""}>
                  {c.npv === null ? "보고값 없음" : money(c.npv)}
                </td>
                <td className={c.hasArchive ? "archive-connected" : ""}>
                  {c.hasArchive
                    ? "준비됨"
                    : c.archiveIssue
                      ? "결과 확인 필요"
                      : "미준비"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {compared.length > 0 ? (
        <>
          <SectionTitle
            title={
              onboarding
                ? "02 · 비교 후 운영할 설비를 선택하세요"
                : "후보를 비교하고 조회할 설비를 선택하세요"
            }
            aside={
              compared.length === 1
                ? "후보 1개를 더 선택해 비교할 수 있습니다"
                : "선택한 2개 후보를 동일 기준으로 비교"
            }
          />
          <CandidateComparison
            candidates={compared}
            chosen={comparison.chosen}
            onChoose={(id) => dispatch({ type: "choose", id })}
          />
          <p className="caption">
            연간 VPP 수입은 투자·유지·교체비 차감 전입니다. NPV는 기존 보고서의
            값을 표시합니다.
          </p>
        </>
      ) : (
        <div className="comparison-empty">
          {rows.length
            ? "위 목록에서 비교할 설비를 선택하세요. 최대 2개 후보의 조건을 나란히 확인할 수 있습니다."
            : scope === "ready"
              ? "운영 시연 가능한 결과가 없습니다. ‘전체 후보’에서 사업성 검토를 이어가세요."
              : "이 선로에 연결된 설비 비교 결과가 없습니다."}
        </div>
      )}
      {candidate ? (
        <CandidateReview
          key={candidate.id}
          candidate={candidate}
          onboarding={onboarding}
          onSelect={onSelect}
        />
      ) : (
        compared.length > 0 && (
          <p className="empty-inline">
            비교표에서 운영할 설비 하나를 선택하면 검토 의견과 다음 단계가
            나타납니다.
          </p>
        )
      )}
    </div>
  );
}
function CandidateComparison({
  candidates,
  chosen,
  onChoose,
}: {
  candidates: Candidate[];
  chosen: string | null;
  onChoose: (id: string) => void;
}) {
  const metrics = [
    { label: "ESS 출력", value: (c: Candidate) => `${number(c.power, 0)} MW` },
    {
      label: "저장 용량",
      value: (c: Candidate) => `${number(c.capacity, 0)} MWh`,
    },
    { label: "연간 VPP 수입", value: (c: Candidate) => money(c.income) },
    {
      label: "20년 NPV",
      value: (c: Candidate) => (c.npv === null ? "보고값 없음" : money(c.npv)),
    },
    {
      label: "운영 시뮬레이션",
      value: (c: Candidate) =>
        c.hasArchive
          ? "준비됨 · 운영계획 이동 가능"
          : c.archiveIssue
            ? "결과 확인 필요 · 사업성 비교 가능"
            : "미준비 · 사업성 비교 가능",
    },
  ];
  return (
    <div className="candidate-comparison">
      <table aria-label="선택한 설비 직접 비교">
        <thead>
          <tr>
            <th scope="col">비교 항목</th>
            {candidates.map((c, index) => (
              <th key={c.id} scope="col">
                <span className="comparison-slot">후보 {index + 1}</span>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric.label}>
              <th scope="row">{metric.label}</th>
              {candidates.map((c) => (
                <td
                  key={c.id}
                  className={
                    metric.label === "20년 NPV" && c.npv !== null && c.npv < 0
                      ? "negative"
                      : ""
                  }
                >
                  {metric.value(c)}
                </td>
              ))}
            </tr>
          ))}
          <tr className="comparison-decision">
            <th scope="row">최종 선택</th>
            {candidates.map((c) => (
              <td key={c.id}>
                <label className={chosen === c.id ? "chosen" : ""}>
                  <input
                    type="radio"
                    name="operating-candidate"
                    aria-label={`${c.label} 운영 대상으로 선택`}
                    checked={chosen === c.id}
                    onChange={() => onChoose(c.id)}
                  />
                  <span>
                    {chosen === c.id ? "이 설비 선택됨" : "이 설비 선택"}
                  </span>
                </label>
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function cAvailabilityMessage(candidate: Candidate) {
  return candidate.archiveIssue
    ? `저장 결과 확인 필요 · ${candidate.archiveIssue} 사업성 비교는 가능하며, 운영계획을 보려면 ‘준비됨’ 후보를 선택하세요.`
    : "운영 시뮬레이션 미준비 · 연간 사업성 결과만 있어 운영계획으로 이동할 수 없습니다. ‘준비됨’ 후보를 선택하세요.";
}

function CandidateReview({
  candidate,
  onboarding,
  onSelect,
}: {
  candidate: Candidate;
  onboarding: boolean;
  onSelect: (id: string) => void;
}) {
  const review = useReview(`resource:${candidate.id}`);
  const [memo, setMemo] = useState(review.value.memo);
  return (
    <>
      <SectionTitle
        title={
          onboarding
            ? "선택한 설비의 검토를 마무리합니다"
            : "선택한 설비의 검토 의견"
        }
        aside={candidate.label}
      />
      <label className="form-label" htmlFor="resource-memo">
        {onboarding ? "선택 이유 · 검토 의견 (선택)" : "재점검 의견"}
      </label>
      <textarea
        id="resource-memo"
        value={memo}
        maxLength={500}
        onChange={(e) => setMemo(e.target.value)}
        placeholder="유지 여부, 확인할 견적과 선로 조건을 기록하세요."
      />
      <div className="bottom-action">
        <button
          className="button"
          onClick={() => review.save({ memo, complete: true })}
        >
          검토 의견 저장
        </button>
        <div className="resource-next">
          <p
            id="operation-availability"
            className={`availability-note ${candidate.hasArchive ? "" : "unavailable"}`}
            role="status"
          >
            {candidate.hasArchive
              ? "운영 시뮬레이션이 준비되어 있습니다. 선택한 설비의 계획·현황·실적을 같은 결과로 이어서 확인합니다."
              : cAvailabilityMessage(candidate)}
          </p>
          <button
            className="button primary"
            disabled={!candidate.hasArchive}
            aria-describedby="operation-availability"
            onClick={() => onSelect(candidate.id)}
          >
            {onboarding
              ? "이 설비로 운영계획 시작"
              : "이 설비의 운영결과로 이동"}
            <ArrowUpRight size={16} />
          </button>
        </div>
      </div>
      <p className="status" role="status">
        {review.error ||
          (review.value.complete
            ? "이 브라우저에 검토 의견을 저장했습니다."
            : "")}
      </p>
      <p className="caption">
        {onboarding
          ? "선택한 설비를 이 브라우저의 시연 운영 대상으로 기억합니다. 저장된 계획과 실적을 연결하며, 실제 설비 설치나 운영계획 생성은 실행하지 않습니다."
          : "조회 대상을 전환합니다. 실제 설비 채택이나 운영 설정 변경은 수행하지 않습니다. 과거 실적은 각 설비의 결과 묶음으로 유지됩니다."}
      </p>
      {!onboarding && !candidate.hasArchive && (
        <details className="details">
          <summary>시간별 결과 연결 방법</summary>
          <p>
            프로젝트 루트에서 기존 계산 CLI를 실행하면 연간 보고값 대조를 통과한
            결과가 저장됩니다.
          </p>
          <code>
            python -m dashboard.distribution_replay --candidate '{candidate.id}'
          </code>
        </details>
      )}
    </>
  );
}
