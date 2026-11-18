import { describe, expect, it } from "vitest";
import { comparisonReducer, type Comparison } from "./comparison";

describe("comparing candidates before deciding", () => {
  it("compares at most two candidates without silently choosing or replacing one", () => {
    let state: Comparison = { ids: [], chosen: null };
    state = comparisonReducer(state, { type: "toggle", id: "A" });
    state = comparisonReducer(state, { type: "toggle", id: "B" });
    expect(state).toEqual({ ids: ["A", "B"], chosen: null });
    expect(comparisonReducer(state, { type: "toggle", id: "C" })).toEqual(
      state,
    );
    expect(comparisonReducer(state, { type: "choose", id: "C" })).toEqual(
      state,
    );
    expect(comparisonReducer(state, { type: "choose", id: "B" }).chosen).toBe(
      "B",
    );
  });
  it("removing the chosen candidate clears the decision before another candidate is added", () => {
    const chosen: Comparison = { ids: ["A", "B"], chosen: "A" };
    expect(comparisonReducer(chosen, { type: "toggle", id: "B" })).toEqual({
      ids: ["A"],
      chosen: "A",
    });
    const removed = comparisonReducer(chosen, { type: "toggle", id: "A" });
    expect(removed).toEqual({ ids: ["B"], chosen: null });
    expect(comparisonReducer(removed, { type: "toggle", id: "C" })).toEqual({
      ids: ["B", "C"],
      chosen: null,
    });
  });
  it("clears both comparison and decision when the line condition changes", () => {
    expect(
      comparisonReducer({ ids: ["A", "B"], chosen: "B" }, { type: "reset" }),
    ).toEqual({ ids: [], chosen: null });
  });
  it("retains only visible candidates and never keeps a hidden decision", () => {
    const state: Comparison = { ids: ["ready", "unready"], chosen: "unready" };
    expect(
      comparisonReducer(state, { type: "retain", ids: ["ready"] }),
    ).toEqual({ ids: ["ready"], chosen: null });
    expect(
      comparisonReducer(
        { ...state, chosen: "ready" },
        { type: "retain", ids: ["ready"] },
      ),
    ).toEqual({ ids: ["ready"], chosen: "ready" });
    expect(
      comparisonReducer(state, { type: "preset", ids: ["A", "A", "B", "C"] }),
    ).toEqual({ ids: ["A", "B"], chosen: null });
  });
});
