from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from air_quality_imputer.pipeline.common import (
    build_parser,
    derive_seed,
    get_model_cfg_from_params,
    load_params,
    set_global_seed,
    to_plain_dict,
)
from air_quality_imputer.tracking.mlflow_utils import MLflowTracker
from air_quality_imputer.training.model_registry import build_model_from_cfg, config_to_dict


SUPPORTED_MODELS = {"transformer", "saits"}


def _load_windows(processed_dir: Path, station: str) -> dict[str, Any]:
    windows_path = processed_dir / station / "windows.npz"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing prepared windows for station {station}: {windows_path}")
    windows = np.load(windows_path)
    required = ["X_train", "X_val_masked", "X_val_ori"]
    for key in required:
        if key not in windows.files:
            raise KeyError(f"Missing key {key!r} in {windows_path}")
    return {
        "X_train": windows["X_train"].astype(np.float32),
        "X_val_masked": windows["X_val_masked"].astype(np.float32),
        "X_val_ori": windows["X_val_ori"].astype(np.float32),
        "windows_path": windows_path,
    }

def _features_from_manifest(processed_dir: Path) -> list[str]:
    manifest_path = processed_dir / "prepare_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values = payload.get("resolved_features") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []
    features = [str(item).strip() for item in values if str(item).strip()]
    return features


def _resolve_training_features(exp: DictConfig, processed_dir: Path, stations: list[str]) -> list[str]:
    configured = [str(item).strip() for item in list(exp.features) if str(item).strip()]
    if configured:
        return configured
    from_manifest = _features_from_manifest(processed_dir)
    if from_manifest:
        return from_manifest
    if not stations:
        raise ValueError("No stations configured and no features available in manifest.")
    sample = _load_windows(processed_dir=processed_dir, station=stations[0])
    width = int(sample["X_train"].shape[2]) if sample["X_train"].ndim == 3 else 0
    if width <= 0:
        raise ValueError("Unable to infer features from prepared windows.")
    return [f"feature_{idx + 1:03d}" for idx in range(width)]


def _apply_transformer_train_mask(model_cfg: DictConfig, train_mask_cfg: dict[str, Any]) -> DictConfig:
    if str(model_cfg.type) != "classic_transformer" or not train_mask_cfg:
        return model_cfg
    mapping = {
        "mode": "train_mask_mode",
        "missing_rate": "train_missing_rate",
        "block_min_len": "train_block_min_len",
        "block_max_len": "train_block_max_len",
        "block_missing_prob": "train_block_missing_prob",
        "feature_block_prob": "train_feature_block_prob",
        "block_no_overlap": "train_block_no_overlap",
    }
    overrides: dict[str, Any] = {}
    for src_key, dst_key in mapping.items():
        if src_key in train_mask_cfg and train_mask_cfg[src_key] is not None:
            overrides[dst_key] = train_mask_cfg[src_key]
    if not overrides:
        return model_cfg
    payload = to_plain_dict(model_cfg)
    params = to_plain_dict(model_cfg.get("params"))
    params.update(overrides)
    payload["params"] = params
    merged = OmegaConf.create(payload)
    if not isinstance(merged, DictConfig):
        raise TypeError("Invalid transformer config after applying training.train_mask.")
    return merged


def _training_common_params(train_cfg: DictConfig) -> dict[str, Any]:
    payload = to_plain_dict(train_cfg)
    payload.pop("train_mask", None)
    payload.pop("shared_validation_mask", None)
    return payload


def _model_train_mask_cfg(train_cfg: DictConfig, model_name: str) -> dict[str, Any]:
    train_mask_cfg = to_plain_dict(train_cfg.get("train_mask"))
    item = train_mask_cfg.get(model_name)
    if isinstance(item, dict):
        return {str(key): value for key, value in item.items()}
    if model_name == "saits":
        return {"mode": "random", "missing_rate": 0.2}
    return {}


def _validate_saits_train_mask(train_mask_cfg: dict[str, Any]) -> None:
    mode = str(train_mask_cfg.get("mode", "random")).lower()
    if mode != "random":
        raise ValueError(f"SAITS supports only random train mask mode, got: {mode}")
    if "missing_rate" in train_mask_cfg:
        missing_rate = float(train_mask_cfg["missing_rate"])
        if abs(missing_rate - 0.2) > 1e-12:
            raise ValueError(
                "Current SAITS backend uses fixed MCAR rate=0.2 during training. "
                f"Configured training.train_mask.saits.missing_rate={missing_rate}"
            )


def run(cfg: DictConfig) -> None:
    exp = cfg.experiment
    train_cfg = cfg.training
    base_seed = int(exp.seed)
    set_global_seed(base_seed)

    processed_dir = Path(cfg.paths.processed_dir)
    models_dir = Path(cfg.paths.models_dir)

    block_size = int(exp.block_size)
    stations = list(exp.stations)
    features = _resolve_training_features(exp, processed_dir=processed_dir, stations=stations)
    if "station" in features:
        raise ValueError("'station' can no longer be used as a model feature. Remove it from experiment.features.")
    n_features = len(features)
    never_mask_features = set(exp.never_mask_features)
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in never_mask_features]
    model_names = list(exp.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unsupported models for V1 pipeline: {unsupported}")

    tracking_cfg = to_plain_dict(cfg.get("tracking"))
    tracking_cfg["dataset_name"] = str(exp.get("dataset", "air_quality"))
    tracker = MLflowTracker(tracking_cfg)
    experiment_params = to_plain_dict(exp)
    experiment_params["resolved_features"] = features
    training_params = _training_common_params(train_cfg)

    for station in stations:
        prepared = _load_windows(processed_dir=processed_dir, station=station)
        station_n_features = int(prepared["X_train"].shape[2]) if prepared["X_train"].ndim == 3 else 0
        if station_n_features != n_features:
            raise ValueError(
                f"Feature mismatch for station {station}: prepared windows have {station_n_features} "
                f"features, but resolved feature list has {n_features}."
            )
        dataset_train: dict[str, Any] = {
            "X": prepared["X_train"],
            "never_mask_feature_indices": never_mask_feature_indices,
        }
        dataset_val: dict[str, Any] = {
            "X": prepared["X_val_masked"],
            "X_ori": prepared["X_val_ori"],
        }

        for model_name in model_names:
            model_cfg = get_model_cfg_from_params(cfg, model_name)
            model_train_mask = _model_train_mask_cfg(train_cfg, model_name)
            model_cfg = _apply_transformer_train_mask(model_cfg, model_train_mask)
            model_params = to_plain_dict(model_cfg)
            model_inner_params = to_plain_dict(model_cfg.get("params"))
            if str(model_cfg.type) == "saits":
                _validate_saits_train_mask(model_train_mask)
            run_seed = derive_seed(base_seed, station=station, model_name=model_name)
            set_global_seed(run_seed)

            model, model_config, model_type, checkpoint_name = build_model_from_cfg(
                model_cfg,
                n_features=n_features,
                block_size=block_size,
            )

            run_name = f"train/{model_name}/{station}/seed-{run_seed}"
            tags = {
                "stage": "train",
                "model": model_name,
                "station": station,
                "seed": run_seed,
                "mask_mode": str(model_train_mask.get("mode", model_inner_params.get("train_mask_mode", "mcar"))),
            }

            with tracker.start_run(run_name=run_name, tags=tags) as active_run:
                run_id = None
                if active_run is not None:
                    run_info = getattr(active_run, "info", None)
                    run_id = getattr(run_info, "run_id", None)
                tracker.log_params(experiment_params, prefix="experiment")
                tracker.log_params(training_params, prefix="training")
                tracker.log_params(model_train_mask, prefix="training.train_mask")
                tracker.log_params(model_params, prefix=f"models.{model_name}")
                tracker.log_params(
                    {"n_features": n_features, "block_size": block_size},
                    prefix="runtime",
                )

                fit_stats = model.fit(
                    dataset_train,
                    validation_data=dataset_val,
                    epochs=int(train_cfg.epochs),
                    batch_size=int(train_cfg.batch_size),
                    initial_lr=float(train_cfg.lr),
                    patience=int(train_cfg.patience),
                    min_delta=float(train_cfg.min_delta),
                )

                out_model_station = models_dir / model_type / station
                out_model_station.mkdir(parents=True, exist_ok=True)
                model_path = out_model_station / checkpoint_name
                model_id = f"{model_name}-{station}-seed-{run_seed}"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config_dict": config_to_dict(model_config),
                        "features": features,
                        "station": station,
                        "model_name": model_name,
                        "model_id": model_id,
                        "seed": run_seed,
                        "train_run_name": run_name,
                        "train_run_id": run_id,
                        "model_type": model_type,
                    },
                    model_path,
                )
                tracker.log_artifact(model_path, artifact_path="model/files")
                tracker.set_tags({"model_id": model_id, "checkpoint_path": str(model_path), "train_run_id": run_id})

                best_loss = fit_stats.get("best_loss") if isinstance(fit_stats, dict) else None
                tracker.log_metrics({"train.loss.best": best_loss})
                print(f"[train] Saved model: {model_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Train imputation models from prepared windows.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
