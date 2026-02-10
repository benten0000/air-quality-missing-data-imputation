from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from air_quality_imputer.training.data_utils import prepare_station_datasets
from air_quality_imputer.training.model_registry import (
    build_model_from_cfg,
    config_to_dict,
    load_model_cfg_by_name,
)


def run(cfg: DictConfig) -> None:
    data_dir = Path(cfg.paths.data_dir)
    models_dir = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    scalers_dir = Path(cfg.paths.scalers_dir)

    requested_features = list(cfg.experiment.features)
    features = [f for f in requested_features if f != "station"]
    if len(features) != len(requested_features):
        print("[INFO] 'station' is handled via station_id embedding and removed from numeric features.")
    never_mask_features = list(cfg.experiment.never_mask_features) if "never_mask_features" in cfg.experiment else []
    never_mask_features_set = set(never_mask_features)
    never_mask_feature_indices = [i for i, f in enumerate(features) if f in never_mask_features_set]
    unknown_never_mask = [f for f in never_mask_features if f not in features]
    if unknown_never_mask:
        print(f"[WARN] never_mask_features not present in experiment.features: {unknown_never_mask}")
    stations = list(cfg.experiment.stations)
    model_names = list(cfg.experiment.models) if "models" in cfg.experiment else [str(cfg.model.type)]
    model_cfgs = [load_model_cfg_by_name(name) for name in model_names]

    for station in stations:
        try:
            print(f"\nProcessing station: {station}")
            datasets = prepare_station_datasets(
                station=station,
                data_dir=data_dir,
                processed_dir=processed_dir,
                scalers_dir=scalers_dir,
                features=features,
                block_size=int(cfg.experiment.block_size),
                step_size=int(cfg.experiment.step_size),
                missing_rate=float(cfg.experiment.missing_rate),
                seed=int(cfg.experiment.seed),
                mask_mode=str(cfg.experiment.mask_mode),
                block_min_len=int(cfg.experiment.block_min_len),
                block_max_len=int(cfg.experiment.block_max_len),
                block_missing_prob=float(cfg.experiment.block_missing_prob)
                if "block_missing_prob" in cfg.experiment
                else None,
                feature_block_prob=float(cfg.experiment.feature_block_prob)
                if "feature_block_prob" in cfg.experiment
                else 0.6,
                block_no_overlap=bool(cfg.experiment.block_no_overlap)
                if "block_no_overlap" in cfg.experiment
                else True,
                never_mask_feature_indices=never_mask_feature_indices,
            )

            dataset_train = {
                "X": datasets["X_train"],
                "station_ids": datasets.get("S_train"),
                "never_mask_feature_indices": never_mask_feature_indices,
            }
            dataset_val = {
                "X": datasets["X_val_masked"],
                "X_ori": datasets["X_val_ori"],
                "station_ids": datasets.get("S_val"),
            }

            for model_cfg in model_cfgs:
                model, model_config, model_type, checkpoint_name = build_model_from_cfg(
                    model_cfg,
                    n_features=len(features),
                    block_size=int(cfg.experiment.block_size),
                    runtime_params={"n_stations": int(datasets.get("n_stations", 1))},
                )
                print(f"Model type: {model_type}")
                print(f"Model params: {model.number_of_params()}")

                model.fit(
                    dataset_train,
                    validation_data=dataset_val,
                    epochs=int(cfg.training.epochs),
                    batch_size=int(cfg.training.batch_size),
                    initial_lr=float(cfg.training.lr),
                    patience=int(cfg.training.patience),
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
                print(f"Saved model: {model_path}")
        except Exception as exc:
            print(f"[ERROR] Station {station} failed: {exc}")

    print("\nTraining finished.")


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
