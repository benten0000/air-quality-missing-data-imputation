import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from air_quality_imputer import exceptions, logger
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


def _requested_model_ref(eval_cfg: dict[str, Any], model_name: str, station: str) -> str | None:
    refs = eval_cfg.get("mlflow_model_refs")
    if not isinstance(refs, dict):
        return None
    ref = refs.get(model_name)
    if isinstance(ref, str):
        return ref.strip() or None
    if isinstance(ref, dict):
        station_ref = ref.get(station)
        if isinstance(station_ref, str):
            return station_ref.strip() or None
    return None


def _resolve_checkpoint_from_mlflow_ref(*, mlflow_ref: str, checkpoint_name: str) -> Path:
    try:
        artifacts_mod = importlib.import_module("mlflow.artifacts")
    except Exception as exc:
        raise exceptions.TrackingError("mlflow is not installed but evaluation.mlflow_model_refs is configured.") from exc
    download_artifacts = getattr(artifacts_mod, "download_artifacts", None)
    if not callable(download_artifacts):
        raise exceptions.TrackingError("mlflow.artifacts.download_artifacts is not available in current MLflow installation.")

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
    raise FileNotFoundError(f"MLflow ref {mlflow_ref!r} resolved to {local_path}, but no checkpoint (*.pt) found.")


def evaluate_station(
    *,
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
    never_mask_features: list[str],
    device: torch.device,
) -> dict[str, Any]:
    windows_path = processed_dir / station / "windows.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing windows dataset: {windows_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    features = [str(item) for item in ckpt["features"]]
    never_mask_set = set(never_mask_features)
    never_mask_idx = [i for i, f in enumerate(features) if f in never_mask_set]

    model = build_model_from_checkpoint(str(ckpt["model_type"]), ckpt["config_dict"])
    model.load_state_dict(ckpt["state_dict"])
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    windows = np.load(windows_path)
    X_test_ori = windows["X_test_ori"].astype(np.float32)
    if int(X_test_ori.shape[2]) != len(features):
        raise exceptions.ValidationError(
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
        never_mask_feature_indices=never_mask_idx,
    )
    eval_mask = np.isnan(X_test_masked) & ~np.isnan(X_test_ori)
    if not eval_mask.any():
        raise exceptions.ValidationError(f"No masked values for station {station}; increase missing_rate")

    X_imputed = model.impute({"X": X_test_masked})
    if X_imputed is None:
        raise exceptions.TrackingError(f"Imputation failed for station {station}")

    err = X_imputed[eval_mask] - X_test_ori[eval_mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return {"station": station, "n_eval": int(eval_mask.sum()), "mae": mae, "rmse": rmse}


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    v = cast(pd.Series, pd.to_numeric(values, errors="coerce")).to_numpy(dtype=float)
    w = cast(pd.Series, pd.to_numeric(weights, errors="coerce")).to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")


def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = cfg.experiment

    stations = list(exp.stations)
    model_names = list(exp.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise exceptions.ModelBuildError(f"Unsupported models for V1 pipeline: {unsupported}")

    models_dir = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    reports_root = Path(cfg.paths.reports_dir)
    metrics_dir = Path(cfg.paths.metrics_dir)

    eval_cfg = to_plain_dict(cfg.get("evaluation"))
    missing_rate = float(eval_cfg["missing_rate"])
    mask_mode = str(eval_cfg["mask_mode"])
    block_min_len = int(eval_cfg["block_min_len"])
    block_max_len = int(eval_cfg["block_max_len"])
    block_missing_prob_raw = eval_cfg["block_missing_prob"]
    block_missing_prob = None if block_missing_prob_raw is None else float(block_missing_prob_raw)
    feature_block_prob = float(eval_cfg["feature_block_prob"])
    block_no_overlap = bool(eval_cfg["block_no_overlap"])
    never_mask_features = [str(x) for x in list(exp.never_mask_features)]
    base_seed = int(exp.seed)

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

        rows: list[dict[str, Any]] = []
        for station in stations:
            mlflow_ref = _requested_model_ref(eval_cfg, model_name=model_name, station=station)
            source = "default"
            model_path = models_dir / model_type / station / checkpoint_name
            if mlflow_ref:
                model_path = _resolve_checkpoint_from_mlflow_ref(
                    mlflow_ref=mlflow_ref,
                    checkpoint_name=checkpoint_name,
                )
                source = "mlflow_ref"

            run_seed = derive_seed(base_seed, station=station, model_name="eval")
            row = evaluate_station(
                station=station,
                model_path=Path(model_path),
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
            rows.append(row)

            with tracker.start_run(
                run_name=f"eval/{model_name}/{station}/seed-{run_seed}",
                tags={
                    "stage": "eval",
                    "model": model_name,
                    "station": station,
                    "seed": run_seed,
                    "mask_mode": mask_mode,
                    "model_source": source,
                },
            ):
                tracker.log_params(eval_cfg, prefix="evaluation")
                tracker.log_params(to_plain_dict(model_cfg), prefix=f"models.{model_name}")
                tracker.log_params(
                    {"checkpoint_path": str(model_path), "mlflow_ref": mlflow_ref, "source": source},
                    prefix="model_source",
                )
                tracker.log_metrics({"eval.mae": row["mae"], "eval.rmse": row["rmse"], "eval.n_eval": row["n_eval"]})

            logger.logger.info(
                f"[eval] {model_name}/{station}: MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, n_eval={row['n_eval']}"
            )

        df = pd.DataFrame(rows)
        df.to_csv(reports_dir / "test_metrics_overall.csv", index=False)
        per_model_overall[model_type] = df

    summary_rows: list[dict[str, Any]] = []
    for model_type, df in per_model_overall.items():
        n_eval = int(cast(pd.Series, pd.to_numeric(df["n_eval"], errors="coerce")).fillna(0).sum()) if not df.empty else 0
        summary_rows.append(
            {
                "model_type": model_type,
                "n_stations": int(df["station"].nunique()) if not df.empty else 0,
                "n_eval": n_eval,
                "mae": _wavg(df["mae"], df["n_eval"]) if not df.empty else float("nan"),
                "rmse": _wavg(df["rmse"], df["n_eval"]) if not df.empty else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)
    summary_path = reports_root / "summary_overall.csv"
    summary_df.to_csv(summary_path, index=False)

    best = None
    if not summary_df.empty:
        valid = summary_df[cast(pd.Series, pd.to_numeric(summary_df["mae"], errors="coerce")).notna()]
        if not valid.empty:
            best = str(cast(pd.DataFrame, valid).sort_values(by=["mae"]).iloc[0]["model_type"])
    metrics_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model_by_mae": best,
        "models": {
            str(row["model_type"]): {
                "mae": float(row["mae"]) if pd.notna(row["mae"]) else None,
                "rmse": float(row["rmse"]) if pd.notna(row["rmse"]) else None,
                "n_eval": int(row["n_eval"]) if pd.notna(row["n_eval"]) else 0,
                "n_stations": int(row["n_stations"]) if pd.notna(row["n_stations"]) else 0,
            }
            for _, row in summary_df.iterrows()
        },
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "model_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    with tracker.start_run(
        run_name=f"eval/summary/all/seed-{base_seed}",
        tags={"stage": "eval_summary", "model": "all", "station": "all", "seed": base_seed, "mask_mode": mask_mode},
    ):
        tracker.log_artifact(summary_path, artifact_path="summary/files")
        tracker.log_artifact(metrics_path, artifact_path="summary/files")

    logger.logger.info(f"[eval] Saved summary: {summary_path}")
    logger.logger.info(f"[eval] Saved metrics: {metrics_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Evaluate imputation models and emit DVC metrics.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
