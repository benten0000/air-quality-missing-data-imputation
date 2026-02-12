from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from air_quality_imputer.pipeline.common import build_parser, load_params, to_plain_dict
from air_quality_imputer.training.data_utils import prepare_station_datasets


def run(cfg: DictConfig) -> None:
    data_dir = Path(cfg.paths.data_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    scalers_dir = Path(cfg.paths.scalers_dir)

    requested_features = list(cfg.experiment.features)
    features = [feature for feature in requested_features if feature != "station"]
    never_mask_features = list(cfg.experiment.never_mask_features)
    never_mask_features_set = set(never_mask_features)
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in never_mask_features_set]

    block_missing_prob_raw = cfg.experiment.block_missing_prob
    block_missing_prob = None if block_missing_prob_raw is None else float(block_missing_prob_raw)
    stations = list(cfg.experiment.stations)

    manifest_rows: list[dict[str, object]] = []
    for station in stations:
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
            block_missing_prob=block_missing_prob,
            feature_block_prob=float(cfg.experiment.feature_block_prob),
            block_no_overlap=bool(cfg.experiment.block_no_overlap),
            never_mask_feature_indices=never_mask_feature_indices,
        )
        manifest_rows.append(
            {
                "station": station,
                "windows_path": str(datasets["windows_path"]),
                "n_stations": int(datasets["n_stations"]),
                "n_train_windows": int(datasets["X_train"].shape[0]),
                "n_val_windows": int(datasets["X_val_ori"].shape[0]),
                "n_test_windows": int(datasets["X_test_ori"].shape[0]),
                "n_features": int(len(datasets["feature_cols"])),
            }
        )
        print(
            f"[prepare] {station}: train={datasets['X_train'].shape[0]}, "
            f"val={datasets['X_val_ori'].shape[0]}, test={datasets['X_test_ori'].shape[0]}"
        )

    manifest_path = processed_dir / "prepare_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "params": to_plain_dict(cfg.experiment),
        "stations": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[prepare] Wrote manifest: {manifest_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Prepare processed datasets for DVC pipeline.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
