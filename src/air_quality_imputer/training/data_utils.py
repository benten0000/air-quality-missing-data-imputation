from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _concat_or_empty(parts: list[np.ndarray], shape: tuple[int, ...], dtype) -> np.ndarray:
    return np.concatenate(parts, axis=0) if parts else np.empty(shape, dtype=dtype)


def _concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def create_sliding_windows(data: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i : i + window_size])
    if not windows:
        raise ValueError("No windows created. Check data length and window_size.")
    return np.stack(windows, axis=0).astype(np.float32)


def train_val_test_split(data: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(data)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def mask_windows(x: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masked = x.copy()
    mask = rng.random(masked.shape) < missing_rate
    masked[mask] = np.nan
    return masked


def mask_windows_blocks(
    x: np.ndarray,
    missing_rate: float,
    seed: int,
    min_block_len: int = 2,
    max_block_len: int = 8,
) -> np.ndarray:
    if min_block_len <= 0 or max_block_len <= 0:
        raise ValueError("Block lengths must be positive.")
    if min_block_len > max_block_len:
        raise ValueError("min_block_len must be <= max_block_len.")

    rng = np.random.default_rng(seed)
    masked = x.copy()
    n_windows, window_size, n_features = masked.shape
    target_missing = int(round(missing_rate * n_windows * window_size * n_features))
    if target_missing <= 0:
        return masked

    mask = np.zeros_like(masked, dtype=bool)
    n_masked = 0
    max_trials = max(1000, target_missing * 20)
    trials = 0

    while n_masked < target_missing and trials < max_trials:
        trials += 1
        w = int(rng.integers(0, n_windows))
        f = int(rng.integers(0, n_features))
        block_len = int(rng.integers(min_block_len, max_block_len + 1))
        start_max = max(1, window_size - block_len + 1)
        start = int(rng.integers(0, start_max))
        end = min(window_size, start + block_len)

        available = np.flatnonzero(~mask[w, start:end, f])
        if len(available) == 0:
            continue
        remaining = target_missing - n_masked
        if remaining <= 0:
            break
        take = min(remaining, len(available))
        chosen_local = rng.choice(available, size=take, replace=False)
        mask[w, start:end, f][chosen_local] = True
        n_masked += int(take)

    if n_masked < target_missing:
        remaining = target_missing - n_masked
        flat = np.flatnonzero(~mask.reshape(-1))
        if len(flat) > 0:
            chosen = rng.choice(flat, size=min(remaining, len(flat)), replace=False)
            mask.reshape(-1)[chosen] = True

    masked[mask] = np.nan
    return masked


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
    if min_block_len <= 0 or max_block_len <= 0:
        raise ValueError("Block lengths must be positive.")
    if min_block_len > max_block_len:
        raise ValueError("min_block_len must be <= max_block_len.")

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
    maskable_feature_mask = np.ones(n_features, dtype=bool)
    if never_mask_feature_indices:
        for idx in never_mask_feature_indices:
            if 0 <= idx < n_features:
                maskable_feature_mask[idx] = False
                mask[:, :, idx] = False
    maskable_observed = observed & maskable_feature_mask.reshape(1, 1, n_features)
    target_missing = int(round(missing_rate * maskable_observed.sum()))
    current_missing = int(mask.sum())
    flat_mask = mask.reshape(-1)
    flat_maskable_observed = maskable_observed.reshape(-1)
    if current_missing > target_missing:
        active = np.flatnonzero(flat_mask)
        if target_missing <= 0:
            flat_mask[:] = False
        elif target_missing < len(active):
            keep_idx = rng.choice(active, size=target_missing, replace=False)
            new_mask = np.zeros_like(flat_mask, dtype=bool)
            new_mask[keep_idx] = True
            flat_mask[:] = new_mask
    elif current_missing < target_missing:
        available = np.flatnonzero(flat_maskable_observed & ~flat_mask)
        need = min(target_missing - current_missing, len(available))
        if need > 0:
            add_idx = rng.choice(available, size=need, replace=False)
            flat_mask[add_idx] = True

    masked = x.copy()
    masked[mask] = np.nan
    return masked


def _inject_station_feature(
    windows: np.ndarray,
    station_id: int,
    n_stations: int,
    station_feature_index: int | None,
) -> np.ndarray:
    if station_feature_index is None:
        return windows
    station_value = 0.0 if n_stations <= 1 else float(station_id) / float(n_stations - 1)
    station_channel = np.full((windows.shape[0], windows.shape[1], 1), station_value, dtype=np.float32)
    left = windows[:, :, :station_feature_index]
    right = windows[:, :, station_feature_index:]
    return np.concatenate([left, station_channel, right], axis=2).astype(np.float32, copy=False)


def _restore_never_mask_features(
    masked: np.ndarray,
    original: np.ndarray,
    never_mask_feature_indices: list[int] | None,
) -> np.ndarray:
    if not never_mask_feature_indices:
        return masked
    valid_indices = [idx for idx in never_mask_feature_indices if 0 <= idx < masked.shape[2]]
    if not valid_indices:
        return masked
    restored = masked.copy()
    restored[:, :, valid_indices] = original[:, :, valid_indices]
    return restored


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
    if mask_mode == "block":
        masked = mask_windows_blocks(
            x,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
        )
    elif mask_mode == "block_feature":
        masked = mask_windows_block_feature(
            x,
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
        masked = mask_windows(x, missing_rate=missing_rate, seed=seed)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")
    return _restore_never_mask_features(masked, x, never_mask_feature_indices=never_mask_feature_indices)


def prepare_station_datasets(
    station: str,
    data_dir: Path,
    processed_dir: Path,
    scalers_dir: Path,
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

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    model_feature_cols = list(features)
    numeric_feature_cols = [c for c in model_feature_cols if c != "station"]
    station_feature_index = model_feature_cols.index("station") if "station" in model_feature_cols else None
    if not numeric_feature_cols:
        raise ValueError("No numeric features provided. Remove 'station' from features or add valid numeric features.")
    required_cols = ["datetime"] + numeric_feature_cols
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Station {station} missing columns: {missing_cols}")

    if "station" in df.columns:
        df["station"] = df["station"].astype(str)
    else:
        df = df.assign(station=str(station))

    group_frames: list[tuple[str, pd.DataFrame]] = []
    for station_name, station_df in df.groupby("station", sort=True):
        cdf = station_df[["datetime"] + numeric_feature_cols].copy()
        cdf["datetime"] = pd.to_datetime(cdf["datetime"], errors="coerce")
        cdf = cdf.dropna(subset=["datetime"]).sort_values("datetime")
        if cdf.empty:
            continue
        cdf[numeric_feature_cols] = cdf[numeric_feature_cols].apply(pd.to_numeric, errors="coerce")
        group_frames.append((str(station_name), cdf))

    if not group_frames:
        raise ValueError(f"No valid station groups found in {file_path}")

    train_frames = []
    val_frames = []
    test_frames = []
    train_arrays = []
    grouped_splits = []
    for station_name, cdf in group_frames:
        data = cdf[numeric_feature_cols].to_numpy(dtype=np.float32)
        dt = cdf["datetime"].to_numpy()
        data_train, data_val, data_test = train_val_test_split(data)
        dt_train, dt_val, dt_test = train_val_test_split(dt)

        for split_data, split_dt, out_list in (
            (data_train, dt_train, train_frames),
            (data_val, dt_val, val_frames),
            (data_test, dt_test, test_frames),
        ):
            out_list.append(
                pd.DataFrame(split_data, columns=numeric_feature_cols).assign(
                    datetime=split_dt,
                    station=station_name,
                )
            )
        if len(data_train) > 0:
            train_arrays.append(data_train)
        grouped_splits.append((station_name, data_train, data_val, data_test))

    if not train_arrays:
        raise ValueError("No train rows available after split; cannot fit scaler.")

    out_data_station = processed_dir / station
    out_data_station.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_arrays, axis=0))

    out_scaler_station = scalers_dir / station
    out_scaler_station.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_scaler_station / "scaler.pkl")

    station_names = sorted(st for st, *_ in grouped_splits)
    station_to_id = {name: i for i, name in enumerate(station_names)}

    X_train_parts = []
    X_val_parts = []
    X_test_parts = []
    S_train_parts = []
    S_val_parts = []
    S_test_parts = []

    def _append_windows(parts: list[np.ndarray], sid_parts: list[np.ndarray], split_data: np.ndarray, sid: int) -> None:
        if len(split_data) < block_size:
            return
        win = create_sliding_windows(
            scaler.transform(split_data),
            window_size=block_size,
            step_size=step_size,
        )
        win = _inject_station_feature(win, sid, len(station_names), station_feature_index)
        parts.append(win)
        sid_parts.append(np.full((len(win),), sid, dtype=np.int64))

    for station_name, data_train, data_val, data_test in grouped_splits:
        sid = station_to_id[station_name]
        _append_windows(X_train_parts, S_train_parts, data_train, sid)
        _append_windows(X_val_parts, S_val_parts, data_val, sid)
        _append_windows(X_test_parts, S_test_parts, data_test, sid)

    n_feat = len(model_feature_cols)
    X_train = _concat_or_empty(X_train_parts, (0, block_size, n_feat), np.float32)
    X_val_ori = _concat_or_empty(X_val_parts, (0, block_size, n_feat), np.float32)
    X_test_ori = _concat_or_empty(X_test_parts, (0, block_size, n_feat), np.float32)
    S_train = _concat_or_empty(S_train_parts, (0,), np.int64)
    S_val = _concat_or_empty(S_val_parts, (0,), np.int64)
    S_test = _concat_or_empty(S_test_parts, (0,), np.int64)

    split_cols = ["station", "datetime"] + numeric_feature_cols
    train_out = _concat_frames(train_frames, split_cols)
    val_out = _concat_frames(val_frames, split_cols)
    test_out = _concat_frames(test_frames, split_cols)
    train_out.to_csv(out_data_station / "train_data.csv", index=False)
    val_out.to_csv(out_data_station / "val_data.csv", index=False)
    test_out.to_csv(out_data_station / "test_data.csv", index=False)

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
        S_train=S_train,
        S_val=S_val,
        S_test=S_test,
        station_names=np.array(station_names, dtype=str),
    )

    return {
        "X_train": X_train,
        "X_val_ori": X_val_ori,
        "X_val_masked": X_val_masked,
        "X_test_ori": X_test_ori,
        "S_train": S_train,
        "S_val": S_val,
        "S_test": S_test,
        "n_stations": len(station_names),
        "feature_cols": model_feature_cols,
        "windows_path": out_data_station / "windows.npz",
    }
