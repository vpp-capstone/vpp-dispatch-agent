import { useEffect, useRef, useState } from "react";
import type { Hour } from "../types";
import { number, time } from "../lib/format";

interface Props {
  hours: Hour[];
  selected: number;
  onSelect: (h: number) => void;
  planOnly?: boolean;
}
export function Timeline({
  hours,
  selected,
  onSelect,
  planOnly = false,
}: Props) {
  const parent = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(700);
  useEffect(() => {
    const element = parent.current!;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(240, entry.contentRect.width)),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const height = 270,
    left = 38,
    right = 14,
    top = 28,
    bottom = 32;
  const plot = width - left - right,
    max =
      Math.max(
        1,
        ...hours.flatMap((h) => [
          h.forecastGeneration,
          h.actualGeneration ?? 0,
          h.plannedSupply,
          h.actualSupply ?? 0,
        ]),
      ) * 1.17;
  const x = (h: number) => left + (h / 24) * plot,
    y = (n: number) => top + (1 - n / max) * (height - top - bottom);
  function line(key: "plannedSupply" | "actualSupply") {
    let path = "",
      previous = false;
    hours.forEach((h, i) => {
      const value = h[key];
      if (
        value === null ||
        (key === "actualSupply" && (i > selected || planOnly))
      ) {
        previous = false;
        return;
      }
      path += `${previous ? "L" : "M"}${x(i)},${y(value)}L${x(i + 1)},${y(value)}`;
      previous = true;
    });
    return path;
  }
  const ticks =
    width < 440 ? [0, 6, 12, 18, 24] : [0, 3, 6, 9, 12, 15, 18, 21, 24];
  return (
    <div className="timeline" ref={parent}>
      <div className="legend">
        <span>
          <i className="generation" />
          발전 예측
        </span>
        <span>
          <i className="plan" />
          확정 공급계획
        </span>
        {!planOnly && (
          <span>
            <i className="actual" />
            실행 공급
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="시간별 발전 예측, 공급계획과 저장된 실행 결과 비교"
      >
        <text x={left} y={14}>
          MWh / 1시간
        </text>
        {!planOnly && (
          <rect
            x={x(selected + 1)}
            y={top}
            width={x(24) - x(selected + 1)}
            height={height - top - bottom}
            fill="var(--surface-subtle)"
          />
        )}
        {[0, 1, 2, 3, 4].map((v) => (
          <g key={v}>
            <line
              x1={left}
              x2={width - right}
              y1={y((max * v) / 4)}
              y2={y((max * v) / 4)}
              className="gridline"
            />
            <text x={left - 8} y={y((max * v) / 4) + 4} textAnchor="end">
              {number((max * v) / 4)}
            </text>
          </g>
        ))}
        {hours.map((h, i) => (
          <rect
            key={i}
            x={x(i) + 1}
            y={y(h.forecastGeneration)}
            width={Math.max(1, plot / 24 - 2)}
            height={y(0) - y(h.forecastGeneration)}
            fill="var(--sun)"
            opacity={i <= selected || planOnly ? 0.65 : 0.25}
          />
        ))}
        <path d={line("plannedSupply")} className="plan-line" />
        <path d={line("actualSupply")} className="actual-line" />
        <line
          x1={x(selected + 0.5)}
          x2={x(selected + 0.5)}
          y1={top}
          y2={height - bottom}
          className="cursor-line"
        />
        {ticks.map((h) => (
          <text
            key={h}
            x={x(h)}
            y={height - 9}
            textAnchor={h === 0 ? "start" : h === 24 ? "end" : "middle"}
          >
            {String(h).padStart(2, "0")}
          </text>
        ))}
        {hours.map((h) => (
          <rect
            key={h.hour}
            x={x(h.hour)}
            y={top}
            width={plot / 24}
            height={height - top - bottom}
            fill="transparent"
            onClick={() => onSelect(h.hour)}
            className="chart-hit"
          >
            <title>
              {time(h.hour)} · 계획 {number(h.plannedSupply, 2)} MWh · 실행{" "}
              {number(h.actualSupply, 2)} MWh
            </title>
          </rect>
        ))}
      </svg>
    </div>
  );
}
