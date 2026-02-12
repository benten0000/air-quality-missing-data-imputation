from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

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
    }


def run(cfg: DictConfig) -> None:
    base_seed = int(cfg.experiment.seed)
    set_global_seed(base_seed)

    processed_dir = Path(cfg.paths.processed_dir)
    models_dir = Path(cfg.paths.models_dir)

    requested_features = list(cfg.experiment.features)
    features = [feature for feature in requested_features if feature != "station"]
    never_mask_features = list(cfg.experiment.never_mask_features)
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in set(never_mask_features)]

    stations = list(cfg.experiment.stations)
    model_names = list(cfg.experiment.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unsupported models for V1 pipeline: {unsupported}")

    tracker = MLflowTracker(to_plain_dict(cfg.get("tracking")))

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
            run_seed = derive_seed(base_seed, station=station, model_name=model_name)
            set_global_seed(run_seed)

            model, model_config, model_type, checkpoint_name = build_model_from_cfg(
                model_cfg,
                n_features=len(features),
                block_size=int(cfg.experiment.block_size),
                runtime_params={"n_stations": int(prepared["n_stations"])},
            )

            run_name = f"train/{model_name}/{station}/seed-{run_seed}"
            tags = {
                "stage": "train",
                "model": model_name,
                "station": station,
                "seed": run_seed,
                "mask_mode": str(cfg.experiment.mask_mode),
            }

            with tracker.start_run(run_name=run_name, tags=tags):
                tracker.log_params(to_plain_dict(cfg.experiment), prefix="experiment")
                tracker.log_params(to_plain_dict(cfg.training), prefix="training")
                tracker.log_params(to_plain_dict(model_cfg), prefix=f"models.{model_name}")
                tracker.log_params({"n_features": len(features), "block_size": int(cfg.experiment.block_size)}, prefix="runtime")

                fit_stats = model.fit(
                    dataset_train,
                    validation_data=dataset_val,
                    epochs=int(cfg.training.epochs),
                    batch_size=int(cfg.training.batch_size),
                    initial_lr=float(cfg.training.lr),
                    patience=int(cfg.training.patience),
                    min_delta=float(cfg.training.min_delta),
                )

                out_model_station = models_dir / model_type / station
                out_model_station.mkdir(parents=True, exist_ok=True)
                model_path = out_model_station / checkpoint_name
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config_dict": config_to_dict(model_config),
                        "features": features,
                        "station": station,
                        "model_type": model_type,
                    },
                    model_path,
                )
                tracker.log_artifact(model_path, artifact_path="model/files")
                logged_model = tracker.log_torch_model(model, artifact_path="model/mlflow")
                tracker.set_tags({"logged_model": str(bool(logged_model)).lower()})

                best_loss = None
                if isinstance(fit_stats, dict):
                    best_loss = fit_stats.get("best_loss")
                tracker.log_metrics({"train.loss.best": best_loss})
                print(f"[train] Saved model: {model_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Train imputation models from prepared windows.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
