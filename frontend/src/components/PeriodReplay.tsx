import { useEffect, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import type { Period, Source } from "../types";
import { useApi } from "../hooks/useApi";
import { money, number } from "../lib/format";
import {
  cumulative,
  dayCharge,
  dayDischarge,
  dayIncome,
  daySupply,
  presetEnd,
  runningIncome,
} from "../lib/period";
import { SectionTitle, Stats } from "./Shared";

const PRESETS = [
  { key: "week", label: "1주" },
  { key: "month", label: "1개월" },
  { key: "quarter", label: "3개월" },
  { key: "year", label: "1년" },
] as const;
const SPEEDS = [1, 3, 7, 15]; // days per second

// Day-by-day replay of the saved result over a chosen period. Nothing is recomputed:
// the totals are sums of the stored hourly result, revealed one day at a time.
export function PeriodReplay({
  source,
  selection,
  dates,
  onOpenDay,
}: {
  source: Source;
  selection: string;
  dates: string[];
  onOpenDay: (day: string) => void;
}) {
  const first = dates[0] ?? "2025-01-01";
  const last = dates.at(-1) ?? "2025-12-31";
  const [range, setRange] = useState({ start: first, end: last });
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(7);
  const params = new URLSearchParams({
    source,
    selection,
    start: range.start,
    end: range.end,
  });
  const result = useApi<Period>(`/api/v1/dashboard/period?${params}`);
  const days = result.data?.days ?? [];
  const lastIndex = Math.max(0, days.length - 1);
  const at = Math.min(cursor, lastIndex);

  useEffect(() => {
    if (!playing || !days.length) return;
    if (at >= lastIndex) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => setCursor(at + 1), 1000 / speed);
    return () => clearTimeout(id);
  }, [playing, at, lastIndex, speed, days.length]);

  const changeRange = (start: string, end: string) => {
    if (!start || !end) return;
    const s = start < first ? first : start > last ? last : start;
    const e = end > last ? last : end < s ? s : end;
    setRange({ start: s, end: e });
    setCursor(0);
    setPlaying(false);
  };
  const togglePlay = () => {
    if (!playing && at >= lastIndex) setCursor(0);
    setPlaying(!playing);
  };

  const data = result.data;
  const today = days[at];
  const totals = cumulative(days, at);
  const executed =
    !!data?.executionAvailable && days.every((d) => d.income !== null);
  const average =
    totals.income === null || !totals.days ? null : totals.income / totals.days;

  return (
    <>
      <SectionTitle
        title="기간 운영 재생"
        aside={`${first} ~ ${last} 저장 결과 중 선택`}
      />
      <div className="period-controls">
        <label>
          시작일
          <input
            type="date"
            value={range.start}
            min={first}
            max={last}
            onChange={(e) => changeRange(e.target.value, range.end)}
          />
        </label>
        <label>
          종료일
          <input
            type="date"
            value={range.end}
            min={range.start}
            max={last}
            onChange={(e) => changeRange(range.start, e.target.value)}
          />
        </label>
        <div className="preset-group" role="group" aria-label="기간 길이">
          {PRESETS.map((p) => {
            const end = presetEnd(range.start, p.key, last);
            return (
              <button
                key={p.key}
                className={`button subtle${range.end === end ? " active" : ""}`}
                onClick={() => changeRange(range.start, end)}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      </div>

      {result.loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle size={20} />
          <p>선택 기간의 저장 결과를 모으는 중입니다.</p>
        </div>
      ) : result.error ? (
        <div className="error-state" role="alert">
          <p>{result.error}</p>
        </div>
      ) : data && today ? (
        <>
          <div className="replay-controls">
            <button
              className="icon-button"
              onClick={() => {
                setPlaying(false);
                setCursor(Math.max(0, at - 1));
              }}
              disabled={at === 0}
              aria-label="이전 날"
            >
              <ChevronLeft size={17} />
            </button>
            <button className="button primary" onClick={togglePlay}>
              {playing ? <Pause size={14} /> : <Play size={14} />}
              {playing
                ? "일시정지"
                : at >= lastIndex
                  ? "처음부터 재생"
                  : "기간 재생"}
            </button>
            <button
              className="icon-button"
              onClick={() => {
                setPlaying(false);
                setCursor(Math.min(lastIndex, at + 1));
              }}
              disabled={at >= lastIndex}
              aria-label="다음 날"
            >
              <ChevronRight size={17} />
            </button>
            <button
              className="icon-button"
              onClick={() => {
                setPlaying(false);
                setCursor(0);
              }}
              aria-label="처음으로"
            >
              <RotateCcw size={15} />
            </button>
            <input
              type="range"
              min="0"
              max={lastIndex}
              value={at}
              onChange={(e) => {
                setPlaying(false);
                setCursor(Number(e.target.value));
              }}
              aria-label="재생 날짜"
            />
            <output>
              {today.date} · {at + 1}/{days.length}일
            </output>
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              aria-label="재생 속도"
            >
              {SPEEDS.map((n) => (
                <option key={n} value={n}>
                  초당 {n}일
                </option>
              ))}
            </select>
          </div>

          <Stats
            items={[
              {
                label: `누적 ${data.incomeLabel}`,
                value: money(totals.income),
                hint: `${range.start} ~ ${today.date} · ${executed ? "실행 백테스트" : "계획 기준"}`,
              },
              {
                label: executed ? "누적 실행 공급" : "누적 공급계획",
                value: `${number(totals.supply, 0)} MWh`,
                hint: `일평균 ${number(totals.supply / totals.days, 1)} MWh`,
              },
              {
                label: "누적 ESS 충전 / 방전",
                value: `${number(totals.charge, 0)} / ${number(totals.discharge, 0)} MWh`,
              },
              {
                label: "일평균 수입",
                value: money(average),
                hint:
                  totals.shortfall === null
                    ? undefined
                    : `누적 공급 부족 ${number(totals.shortfall, 1)} MWh`,
              },
            ]}
          />

          <CumulativeChart days={days} cursor={at} label={data.incomeLabel} />
          <DailyChart days={days} cursor={at} />

          <div className="focus-row">
            <div>
              <strong>
                {today.date} · 하루 {money(dayIncome(today))}
              </strong>
              <p>
                공급 {number(daySupply(today), 1)} MWh · 충전{" "}
                {number(dayCharge(today), 1)} / 방전{" "}
                {number(dayDischarge(today), 1)} MWh · 하루 끝 SOC{" "}
                {number(today.soc ?? today.plannedSoc)}%
              </p>
            </div>
            <button
              className="button"
              onClick={() => {
                setPlaying(false);
                onOpenDay(today.date);
              }}
            >
              이 날 시간별로 보기
            </button>
          </div>
        </>
      ) : null}
    </>
  );
}

const W = 720;
const H = 180;
const PAD = { l: 52, r: 12, t: 12, b: 22 };

function monthTicks(days: { date: string }[]) {
  // Ranges over ~2 months: one tick per month. Shorter: at most ~8 day ticks.
  if (days.length > 62) {
    const ticks: { i: number; label: string }[] = [];
    days.forEach((d, i) => {
      if (i === 0 || d.date.endsWith("-01"))
        ticks.push({ i, label: `${Number(d.date.slice(5, 7))}월` });
    });
    return ticks;
  }
  const step = Math.max(1, Math.ceil(days.length / 8));
  return days
    .map((d, i) => ({
      i,
      label: `${Number(d.date.slice(5, 7))}/${Number(d.date.slice(8))}`,
    }))
    .filter((t) => t.i % step === 0);
}

function CumulativeChart({
  days,
  cursor,
  label,
}: {
  days: Period["days"];
  cursor: number;
  label: string;
}) {
  const running = runningIncome(days);
  const visible = running.slice(0, cursor + 1);
  if (visible.some((v) => v === null))
    return (
      <p className="caption">
        누적 수입을 계산할 수 없는 날이 있어 그래프를 표시하지 않습니다.
      </p>
    );
  const values = running.filter((v): v is number => v !== null);
  // Fixed scale for the whole period so the line grows instead of rescaling each day.
  const max = Math.max(1, ...values);
  const min = Math.min(0, ...values);
  const n = Math.max(1, days.length - 1);
  const x = (i: number) => PAD.l + (i / n) * (W - PAD.l - PAD.r);
  const y = (v: number) =>
    PAD.t + (1 - (v - min) / (max - min)) * (H - PAD.t - PAD.b);
  const points = (visible as number[])
    .map((v, i) => `${x(i)},${y(v)}`)
    .join(" ");
  const area = `${x(0)},${y(0)} ${points} ${x(cursor)},${y(0)}`;
  const current = visible.at(-1) as number;
  return (
    <figure className="period-chart">
      <figcaption>누적 {label}</figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`누적 ${label} 추이`}
      >
        {[0, 0.5, 1].map((f) => {
          const v = min + (max - min) * f;
          return (
            <g key={f}>
              <line
                x1={PAD.l}
                x2={W - PAD.r}
                y1={y(v)}
                y2={y(v)}
                className="grid"
              />
              <text x={PAD.l - 6} y={y(v) + 4} textAnchor="end">
                {money(v)}
              </text>
            </g>
          );
        })}
        {monthTicks(days).map((t) => (
          <text key={t.i} x={x(t.i)} y={H - 6} textAnchor="middle">
            {t.label}
          </text>
        ))}
        <polygon points={area} className="area" />
        <polyline points={points} className="line" />
        <circle cx={x(cursor)} cy={y(current)} r={4} className="dot" />
      </svg>
    </figure>
  );
}

function DailyChart({
  days,
  cursor,
}: {
  days: Period["days"];
  cursor: number;
}) {
  const max = Math.max(
    0.1,
    ...days.map((d) => Math.max(dayCharge(d), dayDischarge(d))),
  );
  const n = days.length;
  const band = (W - PAD.l - PAD.r) / n;
  const mid = (H - PAD.b) / 2 + PAD.t / 2;
  const h = (v: number) => (v / max) * (mid - PAD.t);
  return (
    <figure className="period-chart">
      <figcaption>
        하루 ESS 충전 <span className="swatch charge" /> / 방전{" "}
        <span className="swatch discharge" /> (MWh)
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="하루 충전·방전량">
        <line x1={PAD.l} x2={W - PAD.r} y1={mid} y2={mid} className="grid" />
        <text x={PAD.l - 6} y={PAD.t + 8} textAnchor="end">
          {number(max, 0)}
        </text>
        <text x={PAD.l - 6} y={mid * 2 - PAD.t} textAnchor="end">
          {number(max, 0)}
        </text>
        {days.slice(0, cursor + 1).map((d, i) => (
          <g key={d.date} className={i === cursor ? "today" : undefined}>
            <rect
              x={PAD.l + i * band}
              y={mid - h(dayCharge(d))}
              width={Math.max(1, band * 0.8)}
              height={h(dayCharge(d))}
              className="charge"
            />
            <rect
              x={PAD.l + i * band}
              y={mid}
              width={Math.max(1, band * 0.8)}
              height={h(dayDischarge(d))}
              className="discharge"
            />
          </g>
        ))}
        {monthTicks(days).map((t) => (
          <text key={t.i} x={PAD.l + t.i * band} y={H - 6} textAnchor="start">
            {t.label}
          </text>
        ))}
      </svg>
    </figure>
  );
}
