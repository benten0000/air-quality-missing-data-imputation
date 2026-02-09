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
):
    file_path = data_dir / f"{station}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file for station {station}: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    required_cols = ["datetime"] + features
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Station {station} missing columns: {missing_cols}")

    station_df = df[required_cols].copy()
    station_df[features] = station_df[features].apply(pd.to_numeric, errors="coerce")

    data = station_df[features].to_numpy(dtype=np.float32)
    dt = station_df["datetime"].to_numpy()

    data_train, data_val, data_test = train_val_test_split(data)
    dt_train, dt_val, dt_test = train_val_test_split(dt)

    out_data_station = processed_dir / station
    out_data_station.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(data_train, columns=features).assign(datetime=dt_train).to_csv(out_data_station / "train_data.csv", index=False)
    pd.DataFrame(data_val, columns=features).assign(datetime=dt_val).to_csv(out_data_station / "val_data.csv", index=False)
    pd.DataFrame(data_test, columns=features).assign(datetime=dt_test).to_csv(out_data_station / "test_data.csv", index=False)

    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_train)
    data_val_scaled = scaler.transform(data_val)
    data_test_scaled = scaler.transform(data_test)

    out_scaler_station = scalers_dir / station
    out_scaler_station.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_scaler_station / "scaler.pkl")

    X_train = create_sliding_windows(data_train_scaled, window_size=block_size, step_size=step_size)
    X_val_ori = create_sliding_windows(data_val_scaled, window_size=block_size, step_size=step_size)
    X_test_ori = create_sliding_windows(data_test_scaled, window_size=block_size, step_size=step_size)
    if mask_mode == "block":
        X_val_masked = mask_windows_blocks(
            X_val_ori,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
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
    )

    return {
        "X_train": X_train,
        "X_val_ori": X_val_ori,
        "X_val_masked": X_val_masked,
        "X_test_ori": X_test_ori,
        "windows_path": out_data_station / "windows.npz",
    }
