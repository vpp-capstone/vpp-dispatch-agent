"""Export leakage-aware 2024 validation and frozen 2025 evaluation files.

The 2024 files use the selected improved architecture refit on 2023 labels only.
The 2025 files preserve the already-frozen operational predictions and only
attach valid actual generation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from vpp.generation_forecast.aws_features import load_aws_observations
from vpp.generation_forecast.improved_model import predict_rows, train_selected
from vpp.generation_forecast.infer_improved import delivery_frame
from vpp.generation_forecast.paths import MODEL_DIR, PROCESSED_DIR, REPOSITORY_ROOT
from vpp.generation_forecast.tune_lightgbm import development_frame
from vpp.generation_forecast.wind_regime import AssetBlendedWindModel

KEY_COLUMNS = ["asset_id", "issue_time_utc", "valid_time_utc"]
ACTUAL_COLUMN = "generation_kwh"
ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts"


def _iterations(model: Any) -> int:
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError(f"모델 반복 횟수를 읽을 수 없습니다: {type(model).__name__}")
    return int(booster.current_iteration())


def selection_from_production_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Recover the frozen architecture without reusing its fitted estimators."""
    selection: dict[str, Any] = {
        "weights": dict(bundle["weights"]),
        "production_weights": dict(bundle["weights"]),
        "selected_best_iterations": {
            name: _iterations(model)
            for name, model in bundle["models"].items()
            if hasattr(model, "booster_")
        },
        "selected_model_params": dict(bundle.get("model_params", {})),
    }
    dynamic = bundle["models"].get("dynamic_regime")
    if dynamic is not None:
        asset_model = dynamic if isinstance(dynamic, AssetBlendedWindModel) else None
        shared = asset_model.shared_model if asset_model is not None else dynamic
        dynamic_selection: dict[str, Any] = {
            "regime_boundaries_ms": list(shared.boundaries),
            "expert_weights_by_regime": shared.expert_weights.tolist(),
            "selected_best_iterations": {
                "joint": _iterations(shared.joint_model),
                "experts": {
                    str(regime): _iterations(model)
                    for regime, model in shared.expert_models.items()
                },
            },
            "model_params": dict(bundle["model_params"]["dynamic_regime"]),
        }
        if asset_model is not None:
            dynamic_selection["asset_blend"] = {
                "asset_blend_weights": dict(asset_model.asset_weights),
                "selected_best_iterations": {
                    asset_id: _iterations(model)
                    for asset_id, model in asset_model.asset_models.items()
                },
            }
        selection["dynamic_regime"] = dynamic_selection
        selection["selected_best_iterations"] = {}
    if bundle.get("wind_postprocessing") is not None:
        selection["wind_postprocessing"] = bundle["wind_postprocessing"]
    return selection


def evaluation_frame(result: pd.DataFrame) -> pd.DataFrame:
    """Add the actual label to the operational contract without raw weather fields."""
    if ACTUAL_COLUMN not in result:
        raise ValueError("평가 파일에 결합할 generation_kwh 열이 없습니다.")
    out = delivery_frame(result)
    out[ACTUAL_COLUMN] = pd.to_numeric(result[ACTUAL_COLUMN], errors="coerce")
    return out


def attach_frozen_actuals(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach valid labels while preserving every frozen prediction row and value."""
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("동결 예측 파일에 중복 키가 있습니다.")
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("발전량 라벨에 중복 키가 있습니다.")
    actual = labels[KEY_COLUMNS + [ACTUAL_COLUMN, "label_is_valid"]].copy()
    actual.loc[~actual["label_is_valid"].fillna(False), ACTUAL_COLUMN] = pd.NA
    merged = predictions.merge(
        actual.drop(columns="label_is_valid"),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(predictions):
        raise ValueError("실측 결합 중 동결 예측 행 수가 바뀌었습니다.")
    prediction_columns = [
        column
        for column in predictions
        if column == "improved_generation_kwh"
        or column.startswith("improved_generation_p")
    ]
    for column in prediction_columns:
        if not merged[column].equals(predictions[column]):
            raise ValueError(f"실측 결합 중 예측값이 바뀌었습니다: {column}")
    return merged


def export_2024(
    energy_type: str,
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    aws_observations: pd.DataFrame,
) -> Path:
    frame, category_maps = development_frame(
        data_dir / f"{energy_type}_training_mart.parquet"
    )
    train = frame[frame["dataset_split"].eq("train")].copy()
    validation = frame[frame["dataset_split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError(f"{energy_type}: 2023 학습 또는 2024 검증 행이 없습니다.")
    production = joblib.load(model_dir / f"{energy_type}_improved_production.joblib")
    if category_maps != production["category_maps"]:
        raise ValueError(f"{energy_type}: 운영 번들과 개발 데이터의 범주 맵이 다릅니다.")
    bundle = train_selected(
        energy_type,
        train,
        category_maps,
        selection_from_production_bundle(production),
        aws_observations,
    )
    bundle["training_years"] = [2023]
    bundle["selection_years"] = [2023, 2024]
    result = predict_rows(bundle, validation)
    result["model_version"] = "improved_2023_train_2024_validation"
    result["trained_until"] = "2023-12-31"
    path = output_dir / f"{energy_type}_improved_2024_validation_with_actual.parquet"
    evaluation_frame(result).to_parquet(path, index=False)
    return path


def export_2025(
    energy_type: str,
    data_dir: Path,
    operational_dir: Path,
    output_dir: Path,
) -> Path:
    source = operational_dir / f"{energy_type}_improved_predictions.parquet"
    predictions = pd.read_parquet(source)
    labels = pd.read_parquet(
        data_dir / f"{energy_type}_training_mart.parquet",
        columns=KEY_COLUMNS + [ACTUAL_COLUMN, "label_is_valid"],
    )
    result = attach_frozen_actuals(predictions, labels)
    path = output_dir / f"{energy_type}_improved_2025_with_actual.parquet"
    result.to_parquet(path, index=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--operational", type=Path, default=ARTIFACT_DIR / "2025" / "operational"
    )
    parser.add_argument(
        "--validation-2024-output",
        type=Path,
        default=ARTIFACT_DIR / "2024_validation",
    )
    parser.add_argument(
        "--evaluation-2025-output",
        type=Path,
        default=ARTIFACT_DIR / "2025" / "evaluation",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ARTIFACT_DIR / "generation_evaluation_export_summary.json",
    )
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    args = parser.parse_args()
    args.validation_2024_output.mkdir(parents=True, exist_ok=True)
    args.evaluation_2025_output.mkdir(parents=True, exist_ok=True)
    aws_observations = load_aws_observations(args.aws)
    outputs: list[Path] = []
    for energy_type in ("solar", "wind"):
        outputs.append(
            export_2024(
                energy_type,
                args.data,
                args.models,
                args.validation_2024_output,
                aws_observations,
            )
        )
        outputs.append(
            export_2025(
                energy_type,
                args.data,
                args.operational,
                args.evaluation_2025_output,
            )
        )
    summary = {
        "2024": {
            "purpose": "development validation; selected architecture refit on 2023 labels",
            "training_labels": [2023],
            "actuals_included": True,
        },
        "2025": {
            "purpose": "final frozen evaluation; operational predictions unchanged",
            "training_labels": [2023, 2024],
            "actuals_included": True,
        },
        "files": {
            path.name: {
                "rows": len(pd.read_parquet(path, columns=["asset_id"])),
                "sha256": _sha256(path),
            }
            for path in outputs
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
