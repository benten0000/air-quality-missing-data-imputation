from pathlib import Path
from typing import Hashable, cast

import numpy as np
import pandas as pd


def mask_windows_block_feature(
    x: np.ndarray,
    missing_rate: float,
    seed: int,
    min_block_len: int = 2,
    max_block_len: int = 8,
    block_missing_prob: float | None = None,
    feature_missing_prob: float = 0.6,
    no_overlap: bool = True,
    never_mask_feature_indices: list[int] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(x)
    n_windows, window_size, n_features = x.shape
    p_block = float(missing_rate if block_missing_prob is None else block_missing_prob)
    p_block = min(max(p_block, 0.0), 1.0)
    p_feature = min(max(float(feature_missing_prob), 0.0), 1.0)

    mask = np.zeros_like(x, dtype=bool)
    if no_overlap:
        for w in range(n_windows):
            pos = 0
            while pos < window_size:
                block_len = int(rng.integers(min_block_len, max_block_len + 1))
                end = min(window_size, pos + block_len)
                if rng.random() < p_block:
                    feature_sel = rng.random(n_features) < p_feature
                    if np.any(feature_sel):
                        mask[w, pos:end, feature_sel] = True
                pos = end
    else:
        avg_block_len = max(1.0, (min_block_len + max_block_len) / 2.0)
        n_blocks = max(1, int(round(window_size / avg_block_len)))
        for w in range(n_windows):
            for _ in range(n_blocks):
                block_len = int(rng.integers(min_block_len, max_block_len + 1))
                block_len = min(block_len, window_size)
                start_max = max(1, window_size - block_len + 1)
                start = int(rng.integers(0, start_max))
                end = start + block_len
                if rng.random() < p_block:
                    feature_sel = rng.random(n_features) < p_feature
                    if np.any(feature_sel):
                        mask[w, start:end, feature_sel] = True

    mask &= observed
    maskable_observed = observed.copy()
    if never_mask_feature_indices:
        idx = np.asarray(never_mask_feature_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_features)]
        if idx.size:
            mask[:, :, idx] = False
            maskable_observed[:, :, idx] = False
    target_missing = int(round(missing_rate * maskable_observed.sum()))
    current_missing = int(mask.sum())
    flat_mask = mask.reshape(-1)
    flat_maskable_observed = maskable_observed.reshape(-1)
    if current_missing > target_missing:
        if target_missing <= 0:
            flat_mask[:] = False
        else:
            active = np.flatnonzero(flat_mask)
            if target_missing < len(active):
                flat_mask[:] = False
                flat_mask[rng.choice(active, size=target_missing, replace=False)] = True
    elif current_missing < target_missing:
        available = np.flatnonzero(flat_maskable_observed & ~flat_mask)
        need = min(target_missing - current_missing, len(available))
        if need > 0:
            add_idx = rng.choice(available, size=need, replace=False)
            flat_mask[add_idx] = True

    masked = x.copy()
    masked[mask] = np.nan
    return masked


def mask_windows_by_mode(
    x: np.ndarray,
    *,
    missing_rate: float,
    seed: int,
    mask_mode: str,
    block_min_len: int,
    block_max_len: int,
    block_missing_prob: float | None,
    feature_block_prob: float,
    block_no_overlap: bool,
    never_mask_feature_indices: list[int] | None = None,
) -> np.ndarray:
    if mask_mode == "random":
        rng = np.random.default_rng(seed)
        masked = x.copy()
        masked[rng.random(masked.shape) < missing_rate] = np.nan
    elif mask_mode in {"block", "block_feature"}:
        masked = mask_windows_block_feature(
            x,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
            block_missing_prob=block_missing_prob,
            feature_missing_prob=1.0 if mask_mode == "block" else feature_block_prob,
            no_overlap=block_no_overlap,
            never_mask_feature_indices=never_mask_feature_indices,
        )
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    if never_mask_feature_indices:
        idx = np.asarray(never_mask_feature_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < masked.shape[2])]
        if idx.size:
            masked = masked.copy()
            masked[:, :, idx] = x[:, :, idx]
    return masked


def prepare_station_datasets(
    station: str,
    data_dir: Path,
    processed_dir: Path,
    features: list[str],
    block_size: int,
    step_size: int,
    missing_rate: float,
    seed: int,
    mask_mode: str = "block",
    block_min_len: int = 2,
    block_max_len: int = 8,
    block_missing_prob: float | None = None,
    feature_block_prob: float = 0.6,
    block_no_overlap: bool = True,
    never_mask_feature_indices: list[int] | None = None,
):
    file_path = data_dir / f"{station}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file for station {station}: {file_path}")

    if not features:
        raise ValueError("No features provided. Set experiment.features or dataset.definitions.<name>.feature_names.")
    if "station" in features:
        raise ValueError("'station' can no longer be used as a model feature. Remove it from experiment.features.")

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    if "station" in df.columns:
        unique = df["station"].dropna().astype(str).unique().tolist()
        if len(unique) > 1:
            raise ValueError(
                f"{file_path} contains multiple station values ({len(unique)}). "
                "Split the file into one CSV per station."
            )
    required_cols = ["datetime", *features]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Station {station} missing columns: {missing_cols}")

    df = df[required_cols].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    datetime_subset: list[Hashable] = ["datetime"]
    df = cast(pd.DataFrame, df.dropna(subset=datetime_subset).sort_values("datetime"))  # type: ignore[call-overload]
    if df.empty:
        raise ValueError(f"No valid rows found in {file_path}")
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    values = df[features].to_numpy(dtype=np.float32)
    n = len(values)
    train_end, val_end = int(0.8 * n), int(0.8 * n) + int(0.1 * n)
    train, val, test = values[:train_end], values[train_end:val_end], values[val_end:]
    if len(train) == 0:
        raise ValueError("No train rows available after split; cannot fit scaler.")

    mean, std = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    train_scaled = ((train - mean) / std).astype(np.float32, copy=False)
    val_scaled = ((val - mean) / std).astype(np.float32, copy=False)
    test_scaled = ((test - mean) / std).astype(np.float32, copy=False)

    def _windows(arr: np.ndarray) -> np.ndarray:
        if len(arr) < block_size:
            return np.empty((0, block_size, len(features)), dtype=np.float32)
        return np.lib.stride_tricks.sliding_window_view(arr, block_size, axis=0)[::step_size].astype(
            np.float32, copy=False
        )

    X_train = _windows(train_scaled)
    X_val_ori = _windows(val_scaled)
    X_test_ori = _windows(test_scaled)

    out_data_station = processed_dir / station
    out_data_station.mkdir(parents=True, exist_ok=True)

    X_val_masked = mask_windows_by_mode(
        X_val_ori,
        missing_rate=missing_rate,
        seed=seed,
        mask_mode=mask_mode,
        block_min_len=block_min_len,
        block_max_len=block_max_len,
        block_missing_prob=block_missing_prob,
        feature_block_prob=feature_block_prob,
        block_no_overlap=block_no_overlap,
        never_mask_feature_indices=never_mask_feature_indices,
    )
    np.savez_compressed(
        out_data_station / "windows.npz",
        X_train=X_train,
        X_val_ori=X_val_ori,
        X_val_masked=X_val_masked,
        X_test_ori=X_test_ori,
    )

    return {
        "X_train": X_train,
        "X_val_ori": X_val_ori,
        "X_val_masked": X_val_masked,
        "X_test_ori": X_test_ori,
        "feature_cols": list(features),
        "windows_path": out_data_station / "windows.npz",
    }
