from __future__ import annotations

import json
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from air_quality_imputer.pipeline.common import (
    build_parser,
    derive_seed,
    get_model_cfg_from_params,
    load_params,
    to_plain_dict,
)
from air_quality_imputer.tracking.mlflow_utils import MLflowTracker
from air_quality_imputer.training.data_utils import mask_windows_by_mode
from air_quality_imputer.training.model_registry import build_model_from_checkpoint


SUPPORTED_MODELS = {"transformer", "saits"}
EVAL_SETTING_KEYS = (
    "missing_rate",
    "mask_mode",
    "block_min_len",
    "block_max_len",
    "block_missing_prob",
    "feature_block_prob",
    "block_no_overlap",
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return mae, rmse


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_np = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights_np = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values_np) & np.isfinite(weights_np) & (weights_np > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values_np[valid], weights=weights_np[valid]))


def _eval_settings_from_cfg(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in EVAL_SETTING_KEYS if key not in eval_cfg]
    if missing:
        raise KeyError(f"Missing evaluation settings: {missing}. Set them under evaluation.* in params.")
    return {key: eval_cfg[key] for key in EVAL_SETTING_KEYS}

def _requested_model_ref(eval_cfg: dict[str, Any], model_name: str, station: str) -> str | None:
    values = eval_cfg.get("mlflow_model_refs")
    if not isinstance(values, dict):
        return None
    value = values.get(model_name)
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        station_id = value.get(station)
        if isinstance(station_id, str):
            return station_id.strip() or None
    return None


def _resolve_checkpoint_from_mlflow_ref(
    *,
    mlflow_ref: str,
    checkpoint_name: str,
) -> Path:
    try:
        artifacts_mod = importlib.import_module("mlflow.artifacts")
    except Exception as exc:
        raise RuntimeError("mlflow is not installed but evaluation.mlflow_model_refs is configured.") from exc

    download_artifacts = getattr(artifacts_mod, "download_artifacts", None)
    if not callable(download_artifacts):
        raise RuntimeError("mlflow.artifacts.download_artifacts is not available in current MLflow installation.")
    downloaded = download_artifacts(artifact_uri=mlflow_ref)
    local_path = Path(str(downloaded))
    if local_path.is_file():
        return local_path
    candidate = local_path / checkpoint_name
    if candidate.exists():
        return candidate
    pt_files = sorted(local_path.rglob("*.pt"))
    if pt_files:
        return pt_files[0]
    raise FileNotFoundError(
        f"MLflow ref {mlflow_ref!r} resolved to {local_path}, but no checkpoint file (*.pt) was found."
    )


def _resolve_model_source(
    *,
    models_dir: Path,
    model_name: str,
    model_type: str,
    station: str,
    checkpoint_name: str,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    mlflow_ref = _requested_model_ref(eval_cfg, model_name=model_name, station=station)
    if mlflow_ref:
        model_path = _resolve_checkpoint_from_mlflow_ref(
            mlflow_ref=mlflow_ref,
            checkpoint_name=checkpoint_name,
        )
        return {
            "source": "mlflow_ref",
            "mlflow_ref": mlflow_ref,
            "model_path": model_path,
        }

    return {
        "source": "default",
        "mlflow_ref": None,
        "model_path": models_dir / model_type / station / checkpoint_name,
    }


def evaluate_station(
    station: str,
    model_path: Path,
    processed_dir: Path,
    missing_rate: float,
    seed: int,
    mask_mode: str,
    block_min_len: int,
    block_max_len: int,
    block_missing_prob: float | None,
    feature_block_prob: float,
    block_no_overlap: bool,
    never_mask_features: list[str] | None,
    device: torch.device,
) -> dict[str, Any]:
    windows_path = processed_dir / station / "windows.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing windows dataset: {windows_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_model_type = str(checkpoint["model_type"])
    config = checkpoint["config_dict"]
    features = checkpoint["features"]
    never_mask_features_set = set(never_mask_features or [])
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in never_mask_features_set]

    model = build_model_from_checkpoint(ckpt_model_type, config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    windows = np.load(windows_path)
    X_test_ori = windows["X_test_ori"].astype(np.float32)
    if int(X_test_ori.shape[2]) != int(len(features)):
        raise ValueError(
            f"Checkpoint features ({len(features)}) do not match dataset window features ({X_test_ori.shape[2]}). "
            f"Model={model_path}"
        )
    X_test_masked = mask_windows_by_mode(
        X_test_ori,
        missing_rate=missing_rate,
        seed=seed,
        mask_mode=mask_mode,
        block_min_len=block_min_len,
        block_max_len=block_max_len,
        block_missing_prob=block_missing_prob,
        feature_block_prob=feature_block_prob,
        block_no_overlap=block_no_overlap,
        never_mask_feature_indices=never_mask_feature_indices,
    )

    eval_mask = np.isnan(X_test_masked) & ~np.isnan(X_test_ori)
    if not eval_mask.any():
        raise ValueError(f"No masked values for station {station}; increase missing_rate")

    impute_dataset: dict[str, Any] = {"X": X_test_masked}
    X_imputed = model.impute(impute_dataset)
    if X_imputed is None:
        raise RuntimeError(f"Imputation failed for station {station}")

    overall_mae, overall_rmse = compute_metrics(X_test_ori[eval_mask], X_imputed[eval_mask])
    return {
        "station": station,
        "n_eval": int(eval_mask.sum()),
        "mae": overall_mae,
        "rmse": overall_rmse,
    }


def _to_metrics_json(summary_overall_df: pd.DataFrame) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for _, row in summary_overall_df.iterrows():
        models[str(row["model_type"])] = {
            "mae": float(row["mae"]) if pd.notna(row["mae"]) else None,
            "rmse": float(row["rmse"]) if pd.notna(row["rmse"]) else None,
            "n_eval": int(row["n_eval"]) if pd.notna(row["n_eval"]) else 0,
            "n_stations": int(row["n_stations"]) if pd.notna(row["n_stations"]) else 0,
        }
    best_model = None
    if not summary_overall_df.empty:
        valid = summary_overall_df[pd.to_numeric(summary_overall_df["mae"], errors="coerce").notna()]
        if not valid.empty:
            best_model = str(valid.sort_values("mae").iloc[0]["model_type"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model_by_mae": best_model,
        "models": models,
    }


def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = cfg.experiment

    models_dir = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    reports_root = Path(cfg.paths.reports_dir)
    metrics_dir = Path(cfg.paths.metrics_dir)

    stations = list(exp.stations)
    model_names = list(exp.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unsupported models for V1 pipeline: {unsupported}")

    eval_cfg = to_plain_dict(cfg.get("evaluation"))
    eval_settings = _eval_settings_from_cfg(eval_cfg)
    missing_rate = float(eval_settings["missing_rate"])
    base_seed = int(exp.seed)
    mask_mode = str(eval_settings["mask_mode"])
    block_min_len = int(eval_settings["block_min_len"])
    block_max_len = int(eval_settings["block_max_len"])
    block_missing_prob_raw = eval_settings["block_missing_prob"]
    block_missing_prob = None if block_missing_prob_raw is None else float(block_missing_prob_raw)
    feature_block_prob = float(eval_settings["feature_block_prob"])
    block_no_overlap = bool(eval_settings["block_no_overlap"])
    never_mask_features = list(exp.never_mask_features)

    tracking_cfg = to_plain_dict(cfg.get("tracking"))
    tracking_cfg["dataset_name"] = str(exp.get("dataset", "air_quality"))
    tracker = MLflowTracker(tracking_cfg)
    per_model_overall: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        model_cfg = get_model_cfg_from_params(cfg, model_name)
        model_type = str(model_cfg.type)
        checkpoint_name = str(model_cfg.checkpoint_name)
        reports_dir = reports_root / model_type
        reports_dir.mkdir(parents=True, exist_ok=True)

        overall_rows: list[dict[str, Any]] = []

        for station in stations:
            model_source = _resolve_model_source(
                models_dir=models_dir,
                model_name=model_name,
                model_type=model_type,
                station=station,
                checkpoint_name=checkpoint_name,
                eval_cfg=eval_cfg,
            )
            run_seed = derive_seed(base_seed, station=station, model_name="eval")
            overall_row = evaluate_station(
                station=station,
                model_path=Path(model_source["model_path"]),
                processed_dir=processed_dir,
                missing_rate=missing_rate,
                seed=run_seed,
                mask_mode=mask_mode,
                block_min_len=block_min_len,
                block_max_len=block_max_len,
                block_missing_prob=block_missing_prob,
                feature_block_prob=feature_block_prob,
                block_no_overlap=block_no_overlap,
                never_mask_features=never_mask_features,
                device=device,
            )
            overall_rows.append(overall_row)

            run_name = f"eval/{model_name}/{station}/seed-{run_seed}"
            tags = {
                "stage": "eval",
                "model": model_name,
                "station": station,
                "seed": run_seed,
                "mask_mode": mask_mode,
                "model_source": str(model_source["source"]),
            }

            with tracker.start_run(run_name=run_name, tags=tags):
                model_params = to_plain_dict(model_cfg)
                tracker.log_params(eval_cfg, prefix="evaluation")
                tracker.log_params(model_params, prefix=f"models.{model_name}")
                tracker.log_params(
                    {
                        "checkpoint_path": str(model_source["model_path"]),
                        "mlflow_ref": model_source["mlflow_ref"],
                        "source": model_source["source"],
                    },
                    prefix="model_source",
                )
                tracker.log_metrics(
                    {
                        "eval.mae": overall_row["mae"],
                        "eval.rmse": overall_row["rmse"],
                        "eval.n_eval": overall_row["n_eval"],
                    }
                )

            print(
                f"[eval] {model_name}/{station}: "
                f"MAE={overall_row['mae']:.6f}, RMSE={overall_row['rmse']:.6f}, n_eval={overall_row['n_eval']}"
            )

        overall_df = pd.DataFrame(overall_rows)
        overall_path = reports_dir / "test_metrics_overall.csv"
        overall_df.to_csv(overall_path, index=False)
        per_model_overall[model_type] = overall_df

    summary_overall_rows: list[dict[str, Any]] = []
    for model_type, overall_df in per_model_overall.items():
        if overall_df.empty:
            summary_overall_rows.append(
                {"model_type": model_type, "n_stations": 0, "n_eval": 0, "mae": np.nan, "rmse": np.nan}
            )
            continue
        n_eval = int(pd.to_numeric(overall_df["n_eval"], errors="coerce").fillna(0).sum())
        summary_overall_rows.append(
            {
                "model_type": model_type,
                "n_stations": int(overall_df["station"].nunique()),
                "n_eval": n_eval,
                "mae": weighted_mean(overall_df["mae"], overall_df["n_eval"]),
                "rmse": weighted_mean(overall_df["rmse"], overall_df["n_eval"]),
            }
        )
    summary_overall_df = pd.DataFrame(summary_overall_rows)
    if not summary_overall_df.empty:
        summary_overall_df = summary_overall_df.sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)
    summary_overall_path = reports_root / "summary_overall.csv"
    summary_overall_df.to_csv(summary_overall_path, index=False)

    metrics_payload = _to_metrics_json(summary_overall_df)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "model_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Attach summary artifacts to a single compact run.
    with tracker.start_run(
        run_name=f"eval/summary/all/seed-{base_seed}",
        tags={
            "stage": "eval_summary",
            "model": "all",
            "station": "all",
            "seed": base_seed,
            "mask_mode": mask_mode,
        },
    ):
        tracker.log_artifact(summary_overall_path, artifact_path="summary/files")
        tracker.log_artifact(metrics_path, artifact_path="summary/files")

    print(f"[eval] Saved summary: {summary_overall_path}")
    print(f"[eval] Saved metrics: {metrics_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Evaluate imputation models and emit DVC metrics.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
