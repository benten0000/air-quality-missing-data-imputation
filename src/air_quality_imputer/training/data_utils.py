from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    feature_cols = [c for c in features if c != "station"]
    if not feature_cols:
        raise ValueError("No numeric features provided. Remove 'station' from features or add valid numeric features.")
    required_cols = ["datetime"] + feature_cols
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Station {station} missing columns: {missing_cols}")

    station_column = "station" if "station" in df.columns else None
    group_frames: list[tuple[str, pd.DataFrame]] = []
    if station_column is not None:
        df[station_column] = df[station_column].astype(str)
        for station_name, station_df in df.groupby(station_column, sort=True):
            cdf = station_df[["datetime"] + feature_cols].copy()
            cdf["datetime"] = pd.to_datetime(cdf["datetime"], errors="coerce")
            cdf = cdf.dropna(subset=["datetime"]).sort_values("datetime")
            if cdf.empty:
                continue
            cdf[feature_cols] = cdf[feature_cols].apply(pd.to_numeric, errors="coerce")
            group_frames.append((str(station_name), cdf))
    else:
        cdf = df[["datetime"] + feature_cols].copy()
        cdf["datetime"] = pd.to_datetime(cdf["datetime"], errors="coerce")
        cdf = cdf.dropna(subset=["datetime"]).sort_values("datetime")
        if not cdf.empty:
            cdf[feature_cols] = cdf[feature_cols].apply(pd.to_numeric, errors="coerce")
            group_frames.append((station, cdf))

    if not group_frames:
        raise ValueError(f"No valid station groups found in {file_path}")

    train_frames = []
    val_frames = []
    test_frames = []
    train_arrays = []
    grouped_splits = []
    for station_name, cdf in group_frames:
        data = cdf[feature_cols].to_numpy(dtype=np.float32)
        dt = cdf["datetime"].to_numpy()
        data_train, data_val, data_test = train_val_test_split(data)
        dt_train, dt_val, dt_test = train_val_test_split(dt)

        train_df = pd.DataFrame(data_train, columns=feature_cols).assign(datetime=dt_train, station=station_name)
        val_df = pd.DataFrame(data_val, columns=feature_cols).assign(datetime=dt_val, station=station_name)
        test_df = pd.DataFrame(data_test, columns=feature_cols).assign(datetime=dt_test, station=station_name)
        train_frames.append(train_df)
        val_frames.append(val_df)
        test_frames.append(test_df)
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
    for station_name, data_train, data_val, data_test in grouped_splits:
        sid = station_to_id[station_name]
        if len(data_train) >= block_size:
            train_scaled = scaler.transform(data_train)
            win_train = create_sliding_windows(train_scaled, window_size=block_size, step_size=step_size)
            X_train_parts.append(win_train)
            S_train_parts.append(np.full((len(win_train),), sid, dtype=np.int64))
        if len(data_val) >= block_size:
            val_scaled = scaler.transform(data_val)
            win_val = create_sliding_windows(val_scaled, window_size=block_size, step_size=step_size)
            X_val_parts.append(win_val)
            S_val_parts.append(np.full((len(win_val),), sid, dtype=np.int64))
        if len(data_test) >= block_size:
            test_scaled = scaler.transform(data_test)
            win_test = create_sliding_windows(test_scaled, window_size=block_size, step_size=step_size)
            X_test_parts.append(win_test)
            S_test_parts.append(np.full((len(win_test),), sid, dtype=np.int64))

    n_feat = len(feature_cols)
    X_train = np.concatenate(X_train_parts, axis=0) if X_train_parts else np.empty((0, block_size, n_feat), dtype=np.float32)
    X_val_ori = np.concatenate(X_val_parts, axis=0) if X_val_parts else np.empty((0, block_size, n_feat), dtype=np.float32)
    X_test_ori = np.concatenate(X_test_parts, axis=0) if X_test_parts else np.empty((0, block_size, n_feat), dtype=np.float32)
    S_train = np.concatenate(S_train_parts, axis=0) if S_train_parts else np.empty((0,), dtype=np.int64)
    S_val = np.concatenate(S_val_parts, axis=0) if S_val_parts else np.empty((0,), dtype=np.int64)
    S_test = np.concatenate(S_test_parts, axis=0) if S_test_parts else np.empty((0,), dtype=np.int64)

    train_out = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=["station", "datetime"] + feature_cols)
    val_out = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame(columns=["station", "datetime"] + feature_cols)
    test_out = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(columns=["station", "datetime"] + feature_cols)
    train_out = train_out[["station", "datetime"] + feature_cols]
    val_out = val_out[["station", "datetime"] + feature_cols]
    test_out = test_out[["station", "datetime"] + feature_cols]
    train_out.to_csv(out_data_station / "train_data.csv", index=False)
    val_out.to_csv(out_data_station / "val_data.csv", index=False)
    test_out.to_csv(out_data_station / "test_data.csv", index=False)

    if mask_mode == "block":
        X_val_masked = mask_windows_blocks(
            X_val_ori,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
        )
    elif mask_mode == "block_feature":
        X_val_masked = mask_windows_block_feature(
            X_val_ori,
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
        X_val_masked = mask_windows(X_val_ori, missing_rate=missing_rate, seed=seed)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

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
        "feature_cols": feature_cols,
        "windows_path": out_data_station / "windows.npz",
    }
