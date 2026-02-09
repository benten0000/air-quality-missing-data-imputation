from pathlib import Path

import numpy as np
import pandas as pd


def load_station_csvs(data_dir: str | Path, features: list[str]) -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            continue
        keep = ["datetime"] + [c for c in features if c in df.columns]
        station_df = df[keep].copy()
        station_df["station"] = path.stem
        station_df["datetime"] = pd.to_datetime(station_df["datetime"], errors="coerce")
        station_df = station_df.dropna(subset=["datetime"]).sort_values("datetime")
        frames.append(station_df)

    if not frames:
        raise ValueError("No valid station frames with datetime were loaded")

    return pd.concat(frames, axis=0, ignore_index=True)


def build_windows(frame: pd.DataFrame, features: list[str], block_size: int = 24) -> np.ndarray:
    windows = []
    for _, station_df in frame.groupby("station"):
        x = station_df[features].to_numpy(dtype=np.float32)
        if len(x) < block_size:
            continue
        for i in range(len(x) - block_size + 1):
            windows.append(x[i : i + block_size])

    if not windows:
        raise ValueError("No windows created. Check features and block_size.")

    return np.stack(windows, axis=0)
