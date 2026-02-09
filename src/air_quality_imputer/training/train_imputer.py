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

    features = list(cfg.experiment.features)
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
            )

            dataset_train = {"X": datasets["X_train"]}
            dataset_val = {"X": datasets["X_val_masked"], "X_ori": datasets["X_val_ori"]}

            for model_cfg in model_cfgs:
                model, model_config, model_type, checkpoint_name = build_model_from_cfg(
                    model_cfg,
                    n_features=len(features),
                    block_size=int(cfg.experiment.block_size),
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
