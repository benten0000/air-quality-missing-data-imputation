from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from air_quality_imputer.training.data_utils import mask_windows, mask_windows_block_feature, mask_windows_blocks
from air_quality_imputer.training.model_registry import build_model_from_checkpoint, load_model_cfg_by_name


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return mae, rmse


def evaluate_station(
    station: str,
    model_type: str,
    checkpoint_name: str,
    models_dir: Path,
    processed_dir: Path,
    missing_rate: float,
    seed: int,
    mask_mode: str,
    block_min_len: int,
    block_max_len: int,
    block_missing_prob: float | None,
    feature_block_prob: float,
    block_no_overlap: bool,
    never_mask_features: list[str] | None,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    model_path = models_dir / model_type / station / checkpoint_name
    windows_path = processed_dir / station / "windows.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing windows dataset: {windows_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_model_type = checkpoint.get("model_type", model_type)
    config = checkpoint.get("config_dict", checkpoint.get("config"))
    if config is None:
        raise KeyError(f"Checkpoint at {model_path} is missing 'config_dict' and 'config'")
    features = checkpoint["features"]
    never_mask_features_set = set(never_mask_features or [])
    never_mask_feature_indices = [i for i, f in enumerate(features) if f in never_mask_features_set]

    model = build_model_from_checkpoint(ckpt_model_type, config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    windows = np.load(windows_path)
    X_test_ori = windows["X_test_ori"].astype(np.float32)
    S_test = windows["S_test"].astype(np.int64) if "S_test" in windows.files else None
    if mask_mode == "block":
        X_test_masked = mask_windows_blocks(
            X_test_ori,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
        )
    elif mask_mode == "block_feature":
        X_test_masked = mask_windows_block_feature(
            X_test_ori,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
            block_missing_prob=block_missing_prob,
            feature_missing_prob=feature_block_prob,
            no_overlap=block_no_overlap,
            never_mask_feature_indices=never_mask_feature_indices,
        )
    elif mask_mode == "random":
        X_test_masked = mask_windows(X_test_ori, missing_rate=missing_rate, seed=seed)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    eval_mask = np.isnan(X_test_masked) & ~np.isnan(X_test_ori)
    if not eval_mask.any():
        raise ValueError(f"No masked values for station {station}; increase missing_rate")

    impute_dataset = {"X": X_test_masked}
    if S_test is not None:
        impute_dataset["station_ids"] = S_test
    X_imputed = model.impute(impute_dataset)
    if X_imputed is None:
        raise RuntimeError(f"Imputation failed for station {station}")

    overall_mae, overall_rmse = compute_metrics(X_test_ori[eval_mask], X_imputed[eval_mask])

    per_feature_rows = []
    for i, feature in enumerate(features):
        feature_mask = eval_mask[:, :, i]
        if feature_mask.any():
            mae, rmse = compute_metrics(X_test_ori[:, :, i][feature_mask], X_imputed[:, :, i][feature_mask])
            n_eval = int(feature_mask.sum())
        else:
            mae, rmse, n_eval = np.nan, np.nan, 0
        per_feature_rows.append(
            {
                "station": station,
                "feature": feature,
                "n_eval": n_eval,
                "mae": mae,
                "rmse": rmse,
            }
        )

    overall_row = {
        "station": station,
        "n_eval": int(eval_mask.sum()),
        "mae": overall_mae,
        "rmse": overall_rmse,
    }
    return overall_row, per_feature_rows


def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    stations = list(cfg.experiment.stations)
    model_names = list(cfg.experiment.models) if "models" in cfg.experiment else [str(cfg.model.type)]
    model_cfgs = [load_model_cfg_by_name(name) for name in model_names]
    missing_rate = float(cfg.experiment.missing_rate)
    seed = int(cfg.experiment.seed)
    mask_mode = str(cfg.experiment.mask_mode)
    block_min_len = int(cfg.experiment.block_min_len)
    block_max_len = int(cfg.experiment.block_max_len)
    block_missing_prob = float(cfg.experiment.block_missing_prob) if "block_missing_prob" in cfg.experiment else None
    feature_block_prob = float(cfg.experiment.feature_block_prob) if "feature_block_prob" in cfg.experiment else 0.6
    block_no_overlap = bool(cfg.experiment.block_no_overlap) if "block_no_overlap" in cfg.experiment else True
    never_mask_features = list(cfg.experiment.never_mask_features) if "never_mask_features" in cfg.experiment else []

    for model_cfg in model_cfgs:
        model_type = str(model_cfg.type)
        checkpoint_name = str(model_cfg.checkpoint_name)
        reports_dir = Path(cfg.paths.reports_dir) / model_type
        reports_dir.mkdir(parents=True, exist_ok=True)

        overall_rows = []
        per_feature_rows = []

        for station in stations:
            try:
                overall_row, feature_rows = evaluate_station(
                    station=station,
                    model_type=model_type,
                    checkpoint_name=checkpoint_name,
                    models_dir=models_dir,
                    processed_dir=processed_dir,
                    missing_rate=missing_rate,
                    seed=seed,
                    mask_mode=mask_mode,
                    block_min_len=block_min_len,
                    block_max_len=block_max_len,
                    block_missing_prob=block_missing_prob,
                    feature_block_prob=feature_block_prob,
                    block_no_overlap=block_no_overlap,
                    never_mask_features=never_mask_features,
                    device=device,
                )
                overall_rows.append(overall_row)
                per_feature_rows.extend(feature_rows)
                print(
                    f"[{model_type}] {station}: MAE={overall_row['mae']:.6f}, RMSE={overall_row['rmse']:.6f}, n_eval={overall_row['n_eval']}"
                )
            except Exception as exc:
                print(f"[ERROR][{model_type}] Station {station} failed: {exc}")

        overall_df = pd.DataFrame(overall_rows)
        per_feature_df = pd.DataFrame(per_feature_rows)

        overall_path = reports_dir / "test_metrics_overall.csv"
        per_feature_path = reports_dir / "test_metrics_by_feature.csv"
        overall_df.to_csv(overall_path, index=False)
        per_feature_df.to_csv(per_feature_path, index=False)

        print("\nSaved:")
        print(overall_path)
        print(per_feature_path)


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
