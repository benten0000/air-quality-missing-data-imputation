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
    X_val_masked = mask_windows(X_val_ori, missing_rate=missing_rate, seed=seed)

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
