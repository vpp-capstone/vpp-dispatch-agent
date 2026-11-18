import type { ReactNode } from "react";
import {
  ArrowRight,
  Pause,
  Play,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import type { Dispatch } from "react";
import type { Action, Workspace } from "../lib/workspace";
import { time } from "../lib/format";

export function Stats({
  items,
}: {
  items: { label: string; value: string; hint?: string }[];
}) {
  return (
    <dl className="stats">
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
          {item.hint && <small>{item.hint}</small>}
        </div>
      ))}
    </dl>
  );
}
export function SectionTitle({
  title,
  aside,
}: {
  title: string;
  aside?: ReactNode;
}) {
  return (
    <div className="section-title">
      <h2>{title}</h2>
      {aside && <span>{aside}</span>}
    </div>
  );
}
export function TaskRow({
  when,
  title,
  description,
  action,
  onClick,
}: {
  when: string;
  title: string;
  description: string;
  action: string;
  onClick: () => void;
}) {
  return (
    <div className="task-row">
      <span className="task-time">{when}</span>
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
      <button className="text-button" onClick={onClick}>
        {action}
        <ArrowRight size={14} />
      </button>
    </div>
  );
}
export function ReplayControls({
  state,
  dispatch,
}: {
  state: Workspace;
  dispatch: Dispatch<Action>;
}) {
  return (
    <div className="replay-controls">
      <button
        className="icon-button"
        onClick={() => dispatch({ type: "hour", hour: state.hour - 1 })}
        disabled={state.hour === 0}
        aria-label="이전 시간"
      >
        <ChevronLeft size={17} />
      </button>
      <button
        className="button subtle"
        onClick={() => dispatch({ type: "play" })}
      >
        {state.playing ? <Pause size={14} /> : <Play size={14} />}{" "}
        {state.playing ? "일시정지" : "결과 재생"}
      </button>
      <button
        className="icon-button"
        onClick={() => dispatch({ type: "hour", hour: state.hour + 1 })}
        disabled={state.hour === 23}
        aria-label="다음 시간"
      >
        <ChevronRight size={17} />
      </button>
      <input
        type="range"
        min="0"
        max="23"
        value={state.hour}
        onChange={(e) =>
          dispatch({ type: "hour", hour: Number(e.target.value) })
        }
        aria-label="재생 시간"
      />
      <output>{time(state.hour)}</output>
      <select
        value={state.speed}
        onChange={(e) =>
          dispatch({ type: "speed", speed: Number(e.target.value) })
        }
        aria-label="재생 속도"
      >
        {[1, 2, 4].map((n) => (
          <option key={n} value={n}>
            {n}배속
          </option>
        ))}
      </select>
    </div>
  );
}
