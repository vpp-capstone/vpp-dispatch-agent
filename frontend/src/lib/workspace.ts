import type { Catalog, Source } from "../types";
import type { CustomerSelection } from "./customer";

export interface Workspace {
  source: Source;
  selection: string;
  day: string;
  hour: number;
  playing: boolean;
  speed: number;
}
export type Action =
  | { type: "select"; source: Source; selection: string }
  | { type: "day"; day: string }
  | { type: "hour"; hour: number }
  | { type: "play" }
  | { type: "tick" }
  | { type: "speed"; speed: number };

export function initialWorkspace(
  catalog: Catalog,
  customer?: CustomerSelection | null,
): Workspace {
  const candidate =
    catalog.candidates.find(
      (c) => c.id === "cap_2.50MW|kr_notice_example|2:8" && c.hasArchive,
    ) ?? catalog.candidates.find((c) => c.hasArchive);
  return {
    source: candidate ? "distribution" : "portfolio",
    selection: candidate?.id ?? catalog.portfolio[0]?.id ?? "",
    day: "2025-05-01",
    hour: 13,
    playing: false,
    speed: 4,
    ...customer,
  };
}
export function workspaceReducer(state: Workspace, action: Action): Workspace {
  switch (action.type) {
    case "select":
      return {
        ...state,
        source: action.source,
        selection: action.selection,
        hour: 13,
        playing: false,
      };
    case "day":
      return { ...state, day: action.day, hour: 0, playing: false };
    case "hour":
      return {
        ...state,
        hour: Math.max(0, Math.min(23, action.hour)),
        playing: false,
      };
    case "play":
      return {
        ...state,
        hour: state.hour === 23 ? 0 : state.hour,
        playing: !state.playing,
      };
    case "tick":
      return state.playing
        ? {
            ...state,
            hour: Math.min(23, state.hour + 1),
            playing: state.hour < 22,
          }
        : state;
    case "speed":
      return { ...state, speed: action.speed };
  }
}
