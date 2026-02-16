import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from air_quality_imputer import logger
from air_quality_imputer.pipeline.common import build_parser, load_params, to_plain_dict
from air_quality_imputer.pipeline.dataset_inputs import prepare_dataset_inputs
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
    split_cfg = dataset_info.get("split") if isinstance(dataset_info, dict) else None
    window_cfg = dataset_info.get("window") if isinstance(dataset_info, dict) else None

    def _maybe_int(value: Any) -> int | None:
        if value in (None, "", "null"):
            return None
        return int(value)

    exp_block_raw: Any = exp.get("block_size")
    exp_step_raw: Any = exp.get("step_size")
    rec_block_raw: Any = window_cfg.get("block_size") if isinstance(window_cfg, dict) else None
    rec_step_raw: Any = window_cfg.get("step_size") if isinstance(window_cfg, dict) else None

    exp_block = _maybe_int(exp_block_raw)
    exp_step = _maybe_int(exp_step_raw)
    rec_block = _maybe_int(rec_block_raw)
    rec_step = _maybe_int(rec_step_raw)

    if exp_block is None and rec_block is None:
        raise ValueError("Missing block_size. Set experiment.block_size or dataset.definitions.<name>.window.block_size.")
    if exp_step is None and rec_step is None:
        raise ValueError("Missing step_size. Set experiment.step_size or dataset.definitions.<name>.window.step_size.")

    if exp_block is not None:
        block_size = exp_block
    else:
        assert rec_block is not None
        block_size = rec_block
    if exp_step is not None:
        step_size = exp_step
    else:
        assert rec_step is not None
        step_size = rec_step
    if rec_block is not None and exp_block is not None and exp_block != int(rec_block):
        logger.logger.warning(f"[prepare] block_size={exp_block} differs from dataset window.block_size={rec_block}")
    if rec_step is not None and exp_step is not None and exp_step != int(rec_step):
        logger.logger.warning(f"[prepare] step_size={exp_step} differs from dataset window.step_size={rec_step}")

    # Only used when dataset split config does not specify a split scheme.
    train_split_ratio = float(exp.get("train_split_ratio", 0.8))
    val_split_ratio = float(exp.get("val_split_ratio", 0.1))
    test_split_ratio = float(exp.get("test_split_ratio", 0.1))

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
            block_size=block_size,
            step_size=step_size,
            missing_rate=float(val_mask_cfg.missing_rate),
            seed=int(exp.seed),
            train_split_ratio=train_split_ratio,
            val_split_ratio=val_split_ratio,
            test_split_ratio=test_split_ratio,
            split_cfg=split_cfg if isinstance(split_cfg, dict) else None,
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
        logger.logger.info(
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
    logger.logger.info(f"[prepare] Wrote manifest: {manifest_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Prepare processed datasets for DVC pipeline.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
