import { useEffect, useReducer, useState } from "react";
import {
  ArrowUpRight,
  BatteryMedium,
  CircleHelp,
  RefreshCw,
  Wind,
  Sun,
  LoaderCircle,
} from "lucide-react";
import type { Catalog, Snapshot, Source, Page } from "./types";
import { useApi } from "./hooks/useApi";
import { useRoute } from "./hooks/useRoute";
import { initialWorkspace, workspaceReducer } from "./lib/workspace";
import { number, lineLabel } from "./lib/format";
import { Overview } from "./pages/Overview";
import { Plan } from "./pages/Plan";
import { Settlement } from "./pages/Settlement";
import { Resources } from "./pages/Resources";
import { Welcome } from "./pages/Welcome";
import {
  loadCustomerSelection,
  saveCustomerSelection,
  type CustomerSelection,
} from "./lib/customer";

export default function App() {
  const [revision, setRevision] = useState(0);
  const catalog = useApi<Catalog>("/api/v1/dashboard/catalog", revision);
  if (catalog.error)
    return (
      <div className="startup">
        <h1>결과 연결을 확인해 주세요</h1>
        <p>{catalog.error}</p>
        <p>FastAPI 서버를 실행한 뒤 다시 연결하세요.</p>
        <button className="button" onClick={() => setRevision((r) => r + 1)}>
          다시 연결
        </button>
      </div>
    );
  if (!catalog.data)
    return (
      <div className="startup" role="status">
        <LoaderCircle size={24} />
        <p>운영 결과 목록을 불러오는 중입니다.</p>
      </div>
    );
  return <Workspace catalog={catalog.data} />;
}

function Workspace({ catalog }: { catalog: Catalog }) {
  const [customer, setCustomer] = useState(() =>
    loadCustomerSelection(catalog),
  );
  const [state, dispatch] = useReducer(workspaceReducer, catalog, (value) =>
    initialWorkspace(value, customer),
  );
  const [revision, setRevision] = useState(0);
  const { page, navigate } = useRoute(customer ? "plan" : "welcome");
  const isEntry = page === "welcome" || page === "onboarding";
  const [returnPage, setReturnPage] = useState<Page>("plan");
  const [storageWarning, setStorageWarning] = useState("");
  const params = new URLSearchParams({
    source: state.source,
    selection: state.selection,
    day: state.day,
  });
  const result = useApi<Snapshot>(
    state.selection && !isEntry ? `/api/v1/dashboard/snapshot?${params}` : null,
    revision,
  );
  const data = result.data;
  const [origin, setOrigin] = useState("");
  const to = (next: Page) => {
    if (next === "resources" && page !== "resources") setReturnPage(page);
    navigate(next);
  };
  useEffect(() => {
    if (!state.playing || page !== "overview" || !data) return;
    const id = setInterval(
      () => dispatch({ type: "tick" }),
      1200 / state.speed,
    );
    return () => clearInterval(id);
  }, [state.playing, state.speed, page, data]);
  const startOperating = (selection: CustomerSelection, message: string) => {
    const saved = saveCustomerSelection(selection);
    setCustomer(selection);
    setStorageWarning(
      saved
        ? ""
        : "브라우저 저장이 제한되어 있습니다. 이번에는 계속 이용할 수 있지만 다음 방문에는 다시 선택해야 합니다.",
    );
    dispatch({
      type: "select",
      source: selection.source,
      selection: selection.selection,
    });
    dispatch({ type: "day", day: selection.day });
    setOrigin(message);
    navigate("plan");
  };
  const changeSource = (source: Source) => {
    const selection =
      source === "portfolio"
        ? catalog.portfolio[0]?.id
        : catalog.candidates.find((c) => c.hasArchive)?.id;
    if (selection) {
      setOrigin("");
      dispatch({ type: "select", source, selection });
    }
  };
  const tabs: { page: Page; title: string }[] = [
    { page: "plan", title: "운영계획" },
    { page: "overview", title: "운영 현황" },
    { page: "settlement", title: "실적·정산" },
  ];
  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="#/welcome">
          <span>J</span>JEJU FLOW
        </a>
        <span className="brand-caption">제주 VPP 운영 워크스페이스</span>
        <span className="data-badge">저장 결과 기반</span>
        <div className="header-tools">
          {isEntry ? (
            <span className="entry-label">설비 검토부터 시작하는 운영</span>
          ) : (
            <button className="text-button" onClick={() => to("resources")}>
              추가 사업 검토 <ArrowUpRight size={15} />
            </button>
          )}
        </div>
      </header>
      {!isEntry && page !== "resources" && (
        <nav className="main-nav" aria-label="일상 운영 업무">
          {tabs.map((tab) => (
            <a
              key={tab.page}
              href={`#/${tab.page}`}
              aria-current={tab.page === page ? "page" : undefined}
            >
              {tab.title}
            </a>
          ))}
          <span className="nav-context">{data?.date ?? state.day} · KST</span>
        </nav>
      )}
      <div
        className={`workspace ${isEntry ? "entry-layout" : page === "resources" ? "resource-layout" : ""}`}
      >
        {!isEntry && (
          <aside className="sidebar">
            <div className="eyebrow">운영 대상</div>
            <h2>
              {state.source === "distribution"
                ? "배전망 ESS"
                : "발전 포트폴리오"}
            </h2>
            <label className="form-label" htmlFor="source">
              분석 범위
            </label>
            <select
              id="source"
              value={state.source}
              onChange={(e) => changeSource(e.target.value as Source)}
            >
              <option
                value="distribution"
                disabled={!catalog.candidates.some((c) => c.hasArchive)}
              >
                배전망 ESS · 태양광 5.7 MW
              </option>
              <option value="portfolio" disabled={!catalog.portfolio.length}>
                태양광·풍력 21곳
              </option>
            </select>
            {state.source === "portfolio" && (
              <>
                <label className="form-label" htmlFor="dataset">
                  결과 묶음
                </label>
                <select
                  id="dataset"
                  value={state.selection}
                  onChange={(e) =>
                    dispatch({
                      type: "select",
                      source: "portfolio",
                      selection: e.target.value,
                    })
                  }
                >
                  {catalog.portfolio.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </>
            )}
            <label className="form-label" htmlFor="operating-date">
              운영일
            </label>
            <input
              type="date"
              id="operating-date"
              value={state.day}
              min={data?.dates[0]}
              max={data?.dates.at(-1)}
              onChange={(e) => {
                if (e.target.value)
                  dispatch({ type: "day", day: e.target.value });
              }}
            />
            <div className="side-section">
              <h3>조회 설비</h3>
              <div className="asset-item">
                <Sun size={15} />
                <span>
                  {state.source === "distribution"
                    ? "태양광 5.7 MW"
                    : "태양광 18곳"}
                </span>
              </div>
              {state.source === "portfolio" && (
                <div className="asset-item">
                  <Wind size={15} />
                  <span>풍력 3곳</span>
                </div>
              )}
              <div className="asset-item">
                <BatteryMedium size={15} />
                <span>
                  {data
                    ? `${number(data.resource.power, 0)} MW / ${number(data.resource.capacity, 0)} MWh`
                    : "결과 확인 중"}
                </span>
              </div>
              <p className="caption">
                {data &&
                  (state.source === "distribution"
                    ? lineLabel(data.resource.basis)
                    : `결과 묶음 ${data.resource.basis}`)}
              </p>
            </div>
            <div className="side-section">
              <h3>결과 연결 상태</h3>
              <p className="side-note">
                {data?.executionAvailable
                  ? "계획 → 실행 → 정산 연결"
                  : data
                    ? "계획 조회 · 실행 결과 대기"
                    : "저장 결과 확인 중"}
              </p>
              <button
                className="text-button"
                onClick={() => setRevision((r) => r + 1)}
                disabled={result.loading}
              >
                <RefreshCw size={13} />
                저장 결과 새로 읽기
              </button>
            </div>
            <div className="side-section">
              <button className="side-link" onClick={() => to("resources")}>
                설비·사업성 재점검
                <ArrowUpRight size={14} />
              </button>
              <p className="caption">일상 운영과 분리된 검토 공간</p>
            </div>
            <details className="source-details">
              <summary>
                <CircleHelp size={13} />
                데이터 기준
              </summary>
              <p>
                실시간 관제 화면이 아닌 저장된 운영계획·백테스트의 재생입니다.
              </p>
              <p>
                최적화 설정과 원본 파일은 기존 Python 프로젝트에서 관리합니다.
              </p>
            </details>
          </aside>
        )}
        <main>
          {storageWarning && (
            <p className="origin-note" role="status">
              {storageWarning}
            </p>
          )}
          {page === "welcome" ? (
            <Welcome
              hasSelection={!!customer}
              canResume={!!state.selection}
              onStart={() => navigate("onboarding")}
              onResume={() =>
                startOperating(
                  customer ?? {
                    source: state.source,
                    selection: state.selection,
                    day: state.day,
                  },
                  customer
                    ? "저장해 둔 운영 대상의 계획을 이어서 확인합니다."
                    : "기존 고객 시연 · 예시 설비의 저장된 운영계획입니다.",
                )
              }
            />
          ) : page === "onboarding" ? (
            <Resources
              key="onboarding"
              mode="onboarding"
              catalog={catalog}
              active={state.selection}
              onBack={() => navigate("welcome")}
              onSelect={(id) =>
                startOperating(
                  { source: "distribution", selection: id, day: "2025-05-01" },
                  "시연 운영 대상을 선택했습니다. 이 설비의 저장된 운영계획부터 확인하세요.",
                )
              }
            />
          ) : page === "resources" ? (
            <Resources
              key="review"
              catalog={catalog}
              active={state.selection}
              onBack={() => navigate(returnPage)}
              onSelect={(id) => {
                dispatch({
                  type: "select",
                  source: "distribution",
                  selection: id,
                });
                setOrigin("설비 재점검에서 선택한 대상의 운영 결과입니다.");
                navigate("overview");
              }}
            />
          ) : result.loading ? (
            <div className="loading-state" role="status">
              <LoaderCircle size={22} />
              <h2>같은 설비·날짜의 결과를 연결하고 있습니다</h2>
            </div>
          ) : result.error ? (
            <div className="error-state" role="alert">
              <h1>저장 결과를 표시할 수 없습니다</h1>
              <p>{result.error}</p>
              <button className="button" onClick={() => to("resources")}>
                설비와 연결 상태 확인
              </button>
            </div>
          ) : !data ? (
            <div className="empty-state">
              <h1>연결된 운영 결과가 없습니다</h1>
              <p>
                기존 파이프라인에서 검산 완료 결과를 만든 뒤 다시 읽어 주세요.
              </p>
            </div>
          ) : (
            <>
              {origin && <div className="origin-note">{origin}</div>}
              <div className="context-line">
                <span>
                  {data.source === "distribution"
                    ? `${data.resource.label} · ${lineLabel(data.resource.basis)}`
                    : data.title}
                </span>
                <span>
                  계획·실적 ID {data.provenance.snapshot.slice(0, 8)} ·{" "}
                  {state.day}
                </span>
              </div>
              {page === "overview" && (
                <Overview
                  data={data}
                  state={state}
                  dispatch={dispatch}
                  navigate={to}
                />
              )}
              {page === "plan" && (
                <Plan
                  key={data.id}
                  data={data}
                  hour={state.hour}
                  setHour={(hour) => dispatch({ type: "hour", hour })}
                  navigate={to}
                />
              )}
              {page === "settlement" && (
                <Settlement
                  data={data}
                  setHour={(hour) => dispatch({ type: "hour", hour })}
                  navigate={to}
                />
              )}
              <div className="business-banner">
                <div>
                  <strong>새로운 사업을 검토하시나요?</strong>
                  <p>
                    운영을 이어가면서 다른 설비와 사업성을 별도로 비교하세요.
                  </p>
                </div>
                <button className="text-button" onClick={() => to("resources")}>
                  추가 사업 검토 <ArrowUpRight size={16} />
                </button>
              </div>
              <details className="data-details">
                <summary>계산 범위·결과 출처</summary>
                <p>{data.basis}</p>
                {data.warnings.map((w) => (
                  <p key={w}>{w}</p>
                ))}
                <dl>
                  <dt>결과 ID</dt>
                  <dd>{data.provenance.snapshot}</dd>
                  <dt>보고 자료</dt>
                  <dd>{data.provenance.report}</dd>
                </dl>
              </details>
            </>
          )}
        </main>
      </div>
      <footer className="app-footer">
        <span>JEJU FLOW · 같은 결과 묶음으로 계획과 실적 추적</span>
        <span>설비 제어·입찰 전송 없음</span>
      </footer>
    </div>
  );
}
