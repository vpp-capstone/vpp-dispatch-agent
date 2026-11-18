export interface Comparison {
  ids: string[];
  chosen: string | null;
}

export type ComparisonAction =
  | { type: "toggle"; id: string }
  | { type: "choose"; id: string }
  | { type: "retain"; ids: string[] }
  | { type: "preset"; ids: string[] }
  | { type: "reset" };

export function comparisonReducer(
  state: Comparison,
  action: ComparisonAction,
): Comparison {
  switch (action.type) {
    case "toggle":
      if (state.ids.includes(action.id)) {
        return {
          ids: state.ids.filter((id) => id !== action.id),
          chosen: state.chosen === action.id ? null : state.chosen,
        };
      }
      return state.ids.length < 2
        ? { ...state, ids: [...state.ids, action.id] }
        : state;
    case "choose":
      return state.ids.includes(action.id)
        ? { ...state, chosen: action.id }
        : state;
    case "retain": {
      const ids = state.ids.filter((id) => action.ids.includes(id));
      return {
        ids,
        chosen:
          state.chosen && ids.includes(state.chosen) ? state.chosen : null,
      };
    }
    case "preset":
      return { ids: [...new Set(action.ids)].slice(0, 2), chosen: null };
    case "reset":
      return { ids: [], chosen: null };
  }
}
