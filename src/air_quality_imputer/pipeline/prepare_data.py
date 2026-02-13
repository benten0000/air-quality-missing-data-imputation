from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from air_quality_imputer.pipeline.common import build_parser, load_params, to_plain_dict
from air_quality_imputer.pipeline.dataset_adapters import prepare_dataset_inputs
from air_quality_imputer.training.data_utils import prepare_station_datasets


def run(cfg: DictConfig) -> None:
    exp = cfg.experiment
    training_cfg = cfg.training
    val_mask_cfg = training_cfg.shared_validation_mask
    processed_dir = Path(cfg.paths.processed_dir)
    stations = list(exp.stations)

    requested_features = list(exp.features)
    data_dir, dataset_info, features = prepare_dataset_inputs(
        cfg=cfg,
        stations=stations,
        requested_features=requested_features,
    )
    never_mask_set = set(exp.never_mask_features)
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in never_mask_set]

    block_missing_prob_raw = val_mask_cfg.block_missing_prob
    block_missing_prob = None if block_missing_prob_raw is None else float(block_missing_prob_raw)

    manifest_rows: list[dict[str, object]] = []
    for station in stations:
        datasets = prepare_station_datasets(
            station=station,
            data_dir=data_dir,
            processed_dir=processed_dir,
            features=features,
            block_size=int(exp.block_size),
            step_size=int(exp.step_size),
            missing_rate=float(val_mask_cfg.missing_rate),
            seed=int(exp.seed),
            mask_mode=str(val_mask_cfg.mask_mode),
            block_min_len=int(val_mask_cfg.block_min_len),
            block_max_len=int(val_mask_cfg.block_max_len),
            block_missing_prob=block_missing_prob,
            feature_block_prob=float(val_mask_cfg.feature_block_prob),
            block_no_overlap=bool(val_mask_cfg.block_no_overlap),
            never_mask_feature_indices=never_mask_feature_indices,
        )
        manifest_rows.append(
            {
                "station": station,
                "windows_path": str(datasets["windows_path"]),
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
        "dataset": dataset_info,
        "resolved_features": features,
        "params": to_plain_dict(cfg.experiment),
        "shared_validation_mask": to_plain_dict(val_mask_cfg),
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
