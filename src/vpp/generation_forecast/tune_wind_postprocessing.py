"""Evaluate conservative wind forecast trajectory post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vpp.generation_forecast.paths import RESULTS_DIR
from vpp.generation_forecast.tune_wind_dynamic_regime_ensemble import portfolio_nmae
from vpp.generation_forecast.wind_postprocessing import (
    limit_wind_ramp,
    smooth_wind_prediction,
)

CANDIDATES = {
    "none": ("none", 0.0),
    "median3_10": ("median3", 0.10),
    "median3_25": ("median3", 0.25),
    "mean3_10": ("mean3", 0.10),
    "mean3_25": ("mean3", 0.25),
    "savgol5_10": ("savgol5", 0.10),
    "savgol5_25": ("savgol5", 0.25),
    "ramp_08": ("ramp", 0.08),
    "ramp_12": ("ramp", 0.12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def candidate_prediction(frame: pd.DataFrame, candidate: str):
    method, value = CANDIDATES[candidate]
    base = frame["asset_blend"].to_numpy()
    if method == "none":
        return base
    if method == "ramp":
        return limit_wind_ramp(frame, base, value)
    return smooth_wind_prediction(frame, base, method, value)


def main() -> None:
    args = parse_args()
    folds = pd.read_parquet(args.reports / "wind_asset_model_fold_predictions.parquet")
    holdout = pd.read_parquet(args.reports / "wind_asset_model_2024_predictions.parquet")
    rows = []
    for fold_name, selected in folds.groupby("fold", observed=True):
        for candidate in CANDIDATES:
            rows.append(
                {
                    "candidate": candidate,
                    "fold": fold_name,
                    "portfolio_nmae_pct": portfolio_nmae(
                        selected, candidate_prediction(selected, candidate)
                    ),
                }
            )
    scores = pd.DataFrame(rows)
    ranking = (
        scores.groupby("candidate", observed=True)
        .agg(
            mean_nmae_pct=("portfolio_nmae_pct", "mean"),
            std_nmae_pct=("portfolio_nmae_pct", "std"),
            worst_nmae_pct=("portfolio_nmae_pct", "max"),
        )
        .reset_index()
    )
    ranking["selection_score"] = ranking["mean_nmae_pct"] + 0.25 * ranking["std_nmae_pct"]
    holdout_scores = {
        candidate: portfolio_nmae(
            holdout, candidate_prediction(holdout, candidate)
        )
        for candidate in CANDIDATES
    }
    ranking["holdout_2024_nmae_pct"] = ranking["candidate"].map(holdout_scores)
    baseline = ranking[ranking["candidate"].eq("none")].iloc[0]
    eligible = ranking[
        ranking["selection_score"].le(baseline["selection_score"] + 1e-12)
        & ranking["holdout_2024_nmae_pct"].le(
            baseline["holdout_2024_nmae_pct"] + 1e-12
        )
    ]
    selected = eligible.sort_values(
        ["selection_score", "holdout_2024_nmae_pct"]
    ).iloc[0]
    selected_name = str(selected["candidate"])
    folds["final_prediction"] = candidate_prediction(folds, selected_name)
    holdout["final_prediction"] = candidate_prediction(holdout, selected_name)
    summary = {
        "development_years_only": [2023, 2024],
        "locked_2025_loaded": False,
        "selected_candidate": selected_name,
        "selected_method": CANDIDATES[selected_name],
        "selected_selection_score": float(selected["selection_score"]),
        "holdout_2024_nmae_pct": holdout_scores,
    }
    scores.to_csv(args.reports / "wind_postprocessing_folds.csv", index=False)
    ranking.sort_values("selection_score").to_csv(
        args.reports / "wind_postprocessing_ranking.csv", index=False
    )
    folds.to_parquet(args.reports / "wind_final_fold_predictions.parquet")
    holdout.to_parquet(args.reports / "wind_final_2024_predictions.parquet")
    (args.reports / "wind_postprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.sort_values("selection_score").to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
