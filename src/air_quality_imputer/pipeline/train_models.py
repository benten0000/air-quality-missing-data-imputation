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
    required = ["X_train", "X_val_masked", "X_val_ori", "S_train", "S_val"]
    for key in required:
        if key not in windows.files:
            raise KeyError(f"Missing key {key!r} in {windows_path}")
    station_names = windows["station_names"] if "station_names" in windows.files else np.array([station], dtype=str)
    return {
        "X_train": windows["X_train"].astype(np.float32),
        "X_val_masked": windows["X_val_masked"].astype(np.float32),
        "X_val_ori": windows["X_val_ori"].astype(np.float32),
        "S_train": windows["S_train"].astype(np.int64),
        "S_val": windows["S_val"].astype(np.int64),
        "n_stations": int(len(station_names)),
        "windows_path": windows_path,
    }


def _upsert_model_index(models_dir: Path, entry: dict[str, Any]) -> None:
    index_path = models_dir / "model_index.json"
    index_payload: dict[str, Any] = {"version": 1, "entries": []}
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                index_payload = loaded
        except Exception:
            index_payload = {"version": 1, "entries": []}
    entries = index_payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    model_id = str(entry["model_id"])
    entries = [item for item in entries if not (isinstance(item, dict) and str(item.get("model_id")) == model_id)]
    entries.append(entry)
    index_payload["entries"] = entries
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")


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

    features = list(exp.features)
    n_features = len(features)
    block_size = int(exp.block_size)
    never_mask_features = set(exp.never_mask_features)
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in never_mask_features]

    stations = list(exp.stations)
    model_names = list(exp.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unsupported models for V1 pipeline: {unsupported}")

    tracker = MLflowTracker(to_plain_dict(cfg.get("tracking")))
    experiment_params = to_plain_dict(exp)
    training_params = _training_common_params(train_cfg)

    for station in stations:
        prepared = _load_windows(processed_dir=processed_dir, station=station)
        dataset_train: dict[str, Any] = {
            "X": prepared["X_train"],
            "station_ids": prepared["S_train"],
            "never_mask_feature_indices": never_mask_feature_indices,
        }
        dataset_val: dict[str, Any] = {
            "X": prepared["X_val_masked"],
            "X_ori": prepared["X_val_ori"],
            "station_ids": prepared["S_val"],
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
                runtime_params={"n_stations": int(prepared["n_stations"])},
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
                tracker.log_input_dataset(
                    name=f"prepared-{station}",
                    source=str(prepared["windows_path"]),
                    context="training",
                    preview={
                        "station": station,
                        "n_train_windows": int(prepared["X_train"].shape[0]),
                        "n_features": int(prepared["X_train"].shape[2]) if prepared["X_train"].ndim == 3 else 0,
                    },
                )
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
                _upsert_model_index(
                    models_dir=models_dir,
                    entry={
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_type": model_type,
                        "station": station,
                        "seed": run_seed,
                        "checkpoint_path": str(model_path),
                        "train_run_name": run_name,
                        "train_run_id": run_id,
                    },
                )
                tracker.log_artifact(model_path, artifact_path="model/files")
                mlflow_model_name = model_id
                registered_model_name = (
                    tracker.build_registered_model_name(model_name=model_name, station=station)
                    if tracker.register_models
                    else None
                )
                logged_model = tracker.log_torch_model(
                    model,
                    model_name=mlflow_model_name,
                    registered_model_name=registered_model_name,
                )
                if not logged_model:
                    logged_model = tracker.log_checkpoint_pyfunc_model(
                        checkpoint_path=model_path,
                        model_name=mlflow_model_name,
                        registered_model_name=registered_model_name,
                    )
                tags_payload = {
                    "logged_model": str(bool(logged_model)).lower(),
                    "model_id": model_id,
                    "mlflow_model_name": mlflow_model_name,
                }
                if registered_model_name:
                    tags_payload["registered_model_name"] = registered_model_name
                tracker.set_tags(tags_payload)

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
