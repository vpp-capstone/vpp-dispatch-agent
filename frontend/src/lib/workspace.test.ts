import { describe, it, expect } from "vitest";
import { workspaceReducer, type Workspace } from "./workspace";
import { difference } from "./format";

const state: Workspace = {
  source: "distribution",
  selection: "A",
  day: "2025-05-01",
  hour: 13,
  playing: true,
  speed: 1,
};
describe("shared operating context", () => {
  it("changes the resource atomically and stops replay", () => {
    expect(
      workspaceReducer(state, {
        type: "select",
        source: "portfolio",
        selection: "B",
      }),
    ).toMatchObject({
      source: "portfolio",
      selection: "B",
      day: state.day,
      hour: 13,
      playing: false,
    });
    expect(state.selection).toBe("A");
  });
  it("resets the clock when changing day and never runs beyond the 24-hour result", () => {
    expect(
      workspaceReducer(state, { type: "day", day: "2025-05-02" }),
    ).toMatchObject({ hour: 0, playing: false });
    const ended = workspaceReducer({ ...state, hour: 22 }, { type: "tick" });
    expect(ended).toMatchObject({ hour: 23, playing: false });
    expect(workspaceReducer(ended, { type: "tick" })).toEqual(ended);
    expect(workspaceReducer(ended, { type: "play" })).toMatchObject({
      hour: 0,
      playing: true,
    });
  });
  it("keeps absent actual or planned income absent instead of turning it into zero", () => {
    expect(difference(null, 100)).toBeNull();
    expect(difference(100, null)).toBeNull();
    expect(difference(80, 100)).toBe(-20);
  });
});
