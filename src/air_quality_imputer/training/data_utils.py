from pathlib import Path
from typing import Any, Hashable, cast

import numpy as np
import pandas as pd

from air_quality_imputer import exceptions, logger


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
        observed = ~np.isnan(x)
        masked = x.copy()

        maskable_observed = observed.copy()
        if never_mask_feature_indices:
            idx = np.asarray(never_mask_feature_indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < masked.shape[2])]
            if idx.size:
                maskable_observed[:, :, idx] = False

        target_missing = int(round(float(missing_rate) * maskable_observed.sum()))
        if target_missing > 0:
            available = np.flatnonzero(maskable_observed.reshape(-1))
            if available.size:
                pick = rng.choice(available, size=min(target_missing, int(available.size)), replace=False)
                flat = masked.reshape(-1)
                flat[pick] = np.nan
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
        raise exceptions.ValidationError(f"Unsupported mask_mode: {mask_mode}")

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
    train_split_ratio: float = 0.8,
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    split_cfg: dict[str, Any] | None = None,
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
        raise exceptions.ValidationError("No features provided. Set experiment.features or dataset.definitions.<name>.feature_names.")
    if "station" in features:
        raise exceptions.ValidationError("'station' can no longer be used as a model feature. Remove it from experiment.features.")

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    if "station" in df.columns:
        unique = df["station"].dropna().astype(str).unique().tolist()
        if len(unique) > 1:
            raise exceptions.ValidationError(
                f"{file_path} contains multiple station values ({len(unique)}). "
                "Split the file into one CSV per station."
            )
    required_cols = ["datetime", *features]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise exceptions.ValidationError(f"Station {station} missing columns: {missing_cols}")

    has_series = "series_id" in df.columns
    cols = (["series_id"] if has_series else []) + required_cols
    df = df[cols].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    datetime_subset: list[Hashable] = ["datetime"]
    # For multi-series datasets (e.g. PhysioNet-2012), each series has overlapping timestamps.
    # Keep each series contiguous so window extraction works correctly.
    sort_cols = ["datetime"]
    if "series_id" in df.columns:
        sort_cols = ["series_id", "datetime"]
    df = cast(pd.DataFrame, df.dropna(subset=datetime_subset).sort_values(sort_cols))  # type: ignore[call-overload]
    if df.empty:
        raise exceptions.DatasetLoadError(f"No valid rows found in {file_path}")
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    def _split_date_ranges(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def _parse_int_date(value: int) -> pd.Timestamp | None:
            # Support compact YYYYMMDD / YYYYMMDDHHMM / YYYYMMDDHHMMSS integers in YAML.
            s = str(value)
            try:
                if len(s) == 8:
                    return pd.to_datetime(s, format="%Y%m%d")
                if len(s) == 12:
                    return pd.to_datetime(s, format="%Y%m%d%H%M")
                if len(s) == 14:
                    return pd.to_datetime(s, format="%Y%m%d%H%M%S")
            except Exception:
                return None
            return None

        def _dt(key: str) -> pd.Timestamp | None:
            v = cfg.get(key)
            if v in (None, "", "null"):
                return None
            if isinstance(v, int):
                parsed = _parse_int_date(v)
                if parsed is not None:
                    return parsed
            return pd.to_datetime(v)

        ranges = {
            "train": (_dt("train_start"), _dt("train_end")),
            "val": (_dt("val_start"), _dt("val_end")),
            "test": (_dt("test_start"), _dt("test_end")),
        }
        out: dict[str, pd.DataFrame] = {}
        for name, (start, end) in ranges.items():
            mask = pd.Series(True, index=frame.index)
            if start is not None:
                mask &= frame["datetime"] >= start
            if end is not None:
                mask &= frame["datetime"] <= end
            subset = cast(pd.DataFrame, frame.loc[mask].copy())
            order = np.argsort(cast(pd.Series, subset["datetime"]).to_numpy())
            out[name] = cast(pd.DataFrame, subset.iloc[order])
        return out["train"], out["val"], out["test"]

    def _split_random_series(frame: pd.DataFrame, scheme: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        series_ids = frame["series_id"].dropna().astype(int).unique().tolist()
        if not series_ids:
            raise exceptions.DatasetLoadError("No series_id values found for random_series split.")
        rng = np.random.default_rng(seed)
        rng.shuffle(series_ids)
        n_total = len(series_ids)
        if scheme == "physionet2012":
            n_test = int(round(0.2 * n_total))
            test_ids = set(series_ids[:n_test])
            rest = series_ids[n_test:]
            n_val = int(round(0.2 * len(rest)))
            val_ids = set(rest[:n_val])
            train_ids = set(rest[n_val:])
        else:
            if train_split_ratio + val_split_ratio + test_split_ratio <= 0:
                raise exceptions.ValidationError("Invalid split ratios.")
            n_train = int(round(train_split_ratio * n_total))
            n_val = int(round(val_split_ratio * n_total))
            train_ids = set(series_ids[:n_train])
            val_ids = set(series_ids[n_train : n_train + n_val])
            test_ids = set(series_ids[n_train + n_val :])
        train_f = cast(pd.DataFrame, frame[frame["series_id"].astype(int).isin(train_ids)].copy())
        val_f = cast(pd.DataFrame, frame[frame["series_id"].astype(int).isin(val_ids)].copy())
        test_f = cast(pd.DataFrame, frame[frame["series_id"].astype(int).isin(test_ids)].copy())
        return train_f, val_f, test_f

    scheme = str(split_cfg.get("scheme")).strip().lower() if split_cfg and split_cfg.get("scheme") else None
    if split_cfg:
        train_split_ratio = float(split_cfg.get("train_ratio", train_split_ratio))
        val_split_ratio = float(split_cfg.get("val_ratio", val_split_ratio))
        test_split_ratio = float(split_cfg.get("test_ratio", test_split_ratio))

    has_date_ranges = False
    if split_cfg:
        for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
            if split_cfg.get(key) not in (None, "", "null"):
                has_date_ranges = True
                break

    if has_date_ranges:
        train_df, val_df, test_df = _split_date_ranges(df, split_cfg or {})
    elif has_series:
        # Multi-series datasets (like PhysioNet) are split at the series level.
        train_df, val_df, test_df = _split_random_series(df, scheme=scheme)
    else:
        values = df[features].to_numpy(dtype=np.float32)
        n = len(values)
        train_end, val_end = int(train_split_ratio * n), int(train_split_ratio * n) + int(val_split_ratio * n)
        train_vals, val_vals, test_vals = values[:train_end], values[train_end:val_end], values[val_end:]
        train_df = cast(pd.DataFrame, df.iloc[: len(train_vals)].copy())
        val_df = cast(pd.DataFrame, df.iloc[len(train_vals) : len(train_vals) + len(val_vals)].copy())
        test_df = cast(pd.DataFrame, df.iloc[len(train_vals) + len(val_vals) :].copy())

    if train_df.empty:
        raise exceptions.DatasetLoadError("No train rows available after split; cannot fit scaler.")

    train = train_df[features].to_numpy(dtype=np.float32)
    val = val_df[features].to_numpy(dtype=np.float32)
    test = test_df[features].to_numpy(dtype=np.float32)

    mean, std = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    train_scaled = ((train - mean) / std).astype(np.float32, copy=False)
    val_scaled = ((val - mean) / std).astype(np.float32, copy=False)
    test_scaled = ((test - mean) / std).astype(np.float32, copy=False)

    def _windows_2d(arr: np.ndarray) -> np.ndarray:
        if len(arr) < block_size:
            return np.empty((0, block_size, len(features)), dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(arr, block_size, axis=0)[::step_size]
        # NumPy produces [n_windows, n_features, block_size] here for 2D inputs; transpose to [n_windows, block_size, n_features].
        return np.transpose(windows, (0, 2, 1)).astype(np.float32, copy=False)

    def _windows_by_series(frame: pd.DataFrame, scaled_values: np.ndarray) -> np.ndarray:
        if "series_id" not in frame.columns:
            return _windows_2d(scaled_values)
        out_windows: list[np.ndarray] = []
        # series blocks must align with scaled_values row order
        series = frame["series_id"].to_numpy()
        if len(series) != len(scaled_values):
            raise RuntimeError("series_id alignment mismatch with scaled values.")
        # Find contiguous blocks per series after sorting by datetime above.
        start = 0
        while start < len(series):
            sid = series[start]
            end = start + 1
            while end < len(series) and series[end] == sid:
                end += 1
            w = _windows_2d(scaled_values[start:end])
            if w.size:
                out_windows.append(w)
            start = end
        return np.concatenate(out_windows, axis=0) if out_windows else np.empty((0, block_size, len(features)), dtype=np.float32)

    X_train = _windows_by_series(train_df, train_scaled)
    X_val_ori = _windows_by_series(val_df, val_scaled)
    X_test_ori = _windows_by_series(test_df, test_scaled)

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
