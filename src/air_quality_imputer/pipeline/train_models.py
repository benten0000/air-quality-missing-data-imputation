import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from air_quality_imputer import exceptions, logger
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


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    if not hasattr(model, "parameters"):
        return 0, 0
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return total, trainable


def _load_windows(processed_dir: Path, station: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = processed_dir / station / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared windows for station {station}: {path}")
    w = np.load(path)
    X_train = w["X_train"].astype(np.float32)
    X_val_masked = w["X_val_masked"].astype(np.float32)
    X_val_ori = w["X_val_ori"].astype(np.float32)
    return X_train, X_val_masked, X_val_ori, path


def _features_from_manifest(processed_dir: Path) -> list[str]:
    path = processed_dir / "prepare_manifest.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("resolved_features") if isinstance(payload, dict) else None
    return [str(x).strip() for x in items if str(x).strip()] if isinstance(items, list) else []


def _resolve_features(exp: DictConfig, processed_dir: Path, stations: list[str]) -> list[str]:
    configured = [str(x).strip() for x in list(exp.features) if str(x).strip()]
    if configured:
        return configured
    from_manifest = _features_from_manifest(processed_dir)
    if from_manifest:
        return from_manifest
    if not stations:
        raise exceptions.ValidationError("No stations configured and no features available.")
    X_train, *_ = _load_windows(processed_dir, stations[0])
    if X_train.ndim != 3 or X_train.shape[2] <= 0:
        raise exceptions.ValidationError("Unable to infer features from prepared windows.")
    return [f"feature_{i+1:03d}" for i in range(int(X_train.shape[2]))]


def _resolve_block_size(exp: DictConfig, X_train: np.ndarray) -> int:
    if X_train.ndim != 3 or int(X_train.shape[1]) <= 0:
        raise exceptions.ValidationError("Invalid X_train shape; expected [n_windows, block_size, n_features].")
    derived = int(X_train.shape[1])
    configured = exp.get("block_size")
    if configured is not None and int(configured) != derived:
        logger.logger.warning(
            f"[train] experiment.block_size={configured} differs from prepared windows block_size={derived}; using windows value."
        )
    return derived


def _apply_transformer_train_mask(model_cfg: DictConfig, train_mask: dict[str, Any]) -> DictConfig:
    if str(model_cfg.type) != "classic_transformer" or not train_mask:
        return model_cfg
    mapping = (
        ("mode", "train_mask_mode"),
        ("missing_rate", "train_missing_rate"),
        ("block_min_len", "train_block_min_len"),
        ("block_max_len", "train_block_max_len"),
        ("block_missing_prob", "train_block_missing_prob"),
        ("feature_block_prob", "train_feature_block_prob"),
        ("block_no_overlap", "train_block_no_overlap"),
    )
    params = to_plain_dict(model_cfg.get("params"))
    for src, dst in mapping:
        if src in train_mask and train_mask[src] is not None:
            params[dst] = train_mask[src]
    payload = to_plain_dict(model_cfg)
    payload["params"] = params
    merged = OmegaConf.create(payload)
    if not isinstance(merged, DictConfig):
        raise TypeError("Invalid transformer config after applying training.train_mask.")
    return merged


def _model_train_mask(train_cfg: DictConfig, model_name: str) -> dict[str, Any]:
    cfg = to_plain_dict(train_cfg.get("train_mask"))
    value = cfg.get(model_name)
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {"mode": "random"} if model_name == "saits" else {}


def _validate_saits_mask(train_mask: dict[str, Any]) -> None:
    if str(train_mask.get("mode", "random")).lower() != "random":
        raise exceptions.ValidationError("SAITS supports only random train mask mode.")


def run(cfg: DictConfig) -> None:
    exp = cfg.experiment
    train_cfg = cfg.training
    base_seed = int(exp.seed)
    set_global_seed(base_seed)

    stations = list(exp.stations)
    model_names = list(exp.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise exceptions.ModelBuildError(f"Unsupported models for V1 pipeline: {unsupported}")

    processed_dir = Path(cfg.paths.processed_dir)
    models_dir = Path(cfg.paths.models_dir)
    features = _resolve_features(exp, processed_dir=processed_dir, stations=stations)
    if "station" in features:
        raise exceptions.ValidationError("'station' can no longer be used as a model feature. Remove it from experiment.features.")
    n_features = len(features)
    never_mask_set = set(exp.never_mask_features)
    never_mask_idx = [i for i, f in enumerate(features) if f in never_mask_set]

    tracking_cfg = to_plain_dict(cfg.get("tracking"))
    tracking_cfg["dataset_name"] = str(exp.get("dataset", "air_quality"))
    tracker = MLflowTracker(tracking_cfg)

    for station in stations:
        X_train, X_val_masked, X_val_ori, windows_path = _load_windows(processed_dir, station)
        if X_train.ndim != 3 or int(X_train.shape[2]) != n_features:
            raise exceptions.ValidationError(f"Feature mismatch for station {station}.")
        block_size = _resolve_block_size(exp, X_train)
        train_set = {"X": X_train, "never_mask_feature_indices": never_mask_idx}
        val_set = {"X": X_val_masked, "X_ori": X_val_ori}

        for model_name in model_names:
            model_cfg = get_model_cfg_from_params(cfg, model_name)
            train_mask = _model_train_mask(train_cfg, model_name)
            if str(model_cfg.type) == "saits":
                _validate_saits_mask(train_mask)
            model_cfg = _apply_transformer_train_mask(model_cfg, train_mask)

            seed = derive_seed(base_seed, station=station, model_name=model_name)
            set_global_seed(seed)

            # Make SAITS MCAR masking rate configurable and logged; PyPOTS uses DatasetForSAITS(rate=...).
            runtime_params = None
            if str(model_cfg.type) == "saits":
                runtime_params = {"train_missing_rate": float(train_mask.get("missing_rate", 0.2) or 0.2)}

            model, model_config, model_type, checkpoint_name = build_model_from_cfg(
                model_cfg,
                n_features=n_features,
                block_size=block_size,
                runtime_params=runtime_params,
            )
            total_params: int | None = None
            trainable_params: int | None = None
            if model_name == "transformer":
                total_params, trainable_params = _count_parameters(model)
                logger.logger.info(
                    f"[train] transformer params (station={station}): total={total_params:,}, trainable={trainable_params:,}"
                )

            run_name = f"train/{model_name}/{station}/seed-{seed}"
            with tracker.start_run(run_name=run_name, tags={"stage": "train", "model": model_name, "station": station, "seed": seed}):
                tracker.log_params(to_plain_dict(exp) | {"resolved_features": features}, prefix="experiment")
                tracker.log_params(to_plain_dict(train_cfg), prefix="training")
                tracker.log_params(train_mask, prefix="training.train_mask")
                tracker.log_params(to_plain_dict(model_cfg), prefix=f"models.{model_name}")
                runtime_logged = {"n_features": n_features, "block_size": block_size, "windows": str(windows_path)}
                if total_params is not None and trainable_params is not None:
                    runtime_logged["model_params_total"] = total_params
                    runtime_logged["model_params_trainable"] = trainable_params
                tracker.log_params(runtime_logged, prefix="runtime")

                fit_stats = model.fit(
                    train_set,
                    validation_data=val_set,
                    epochs=int(train_cfg.epochs),
                    batch_size=int(train_cfg.batch_size),
                    initial_lr=None,
                    patience=int(train_cfg.patience),
                    min_delta=float(train_cfg.min_delta),
                )

                out_dir = models_dir / model_type / station
                out_dir.mkdir(parents=True, exist_ok=True)
                model_path = out_dir / str(checkpoint_name)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config_dict": config_to_dict(model_config),
                        "features": features,
                        "model_type": model_type,
                        "seed": seed,
                    },
                    model_path,
                )
                tracker.log_artifact(model_path, artifact_path="model/files")
                # DagsHub UI "Models" column is reliably populated by an MLflow model artifact (MLmodel).
                tracker.log_model_stub(
                    local_path=model_path,
                    model_name=model_name,
                )
                best_loss = fit_stats.get("best_loss") if isinstance(fit_stats, dict) else None
                tracker.log_metrics({"train.loss.best": best_loss})
                tracker.set_tags({"checkpoint_path": str(model_path)})
                logger.logger.info(f"[train] Saved model: {model_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Train imputation models from prepared windows.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
