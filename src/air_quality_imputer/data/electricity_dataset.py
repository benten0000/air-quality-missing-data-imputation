import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ELECTRICITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
DEFAULT_RAW_NAME = "LD2011_2014.txt"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, out_path.open("wb") as f:  # nosec: B310
        while chunk := r.read(1024 * 1024):
            f.write(chunk)


def _extract(zip_path: Path, member_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z, z.open(member_name) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _choose_clients(header_cols: list[str], clients: list[str] | None, start_client: int, n_clients: int) -> list[str]:
    mt_cols = [c for c in header_cols if c.startswith("MT_")]
    if not mt_cols:
        raise ValueError("No MT_* client columns found in Electricity file header.")
    if clients:
        missing = [c for c in clients if c not in mt_cols]
        if missing:
            raise ValueError(f"Unknown Electricity clients: {missing}")
        return clients
    if start_client < 1:
        raise ValueError("--start-client must be >= 1")
    if n_clients <= 0:
        raise ValueError("--n-clients must be > 0")
    chosen = mt_cols[start_client - 1 : start_client - 1 + n_clients]
    if not chosen:
        raise ValueError("No Electricity client columns selected.")
    return chosen


def _load_electricity_frame(
    *,
    cache_dir: Path,
    url: str,
    force_download: bool,
    force_extract: bool,
    clients: list[str] | None,
    start_client: int,
    n_clients: int,
    resample_frequency: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str], Path, Path]:
    zip_path = cache_dir / "LD2011_2014.txt.zip"
    raw_txt_path = cache_dir / DEFAULT_RAW_NAME

    if force_download or not zip_path.exists():
        print(f"[electricity] Downloading source archive to {zip_path}")
        _download(url, zip_path)
    else:
        print(f"[electricity] Using cached archive: {zip_path}")

    if force_extract or not raw_txt_path.exists():
        print(f"[electricity] Extracting {DEFAULT_RAW_NAME} to {raw_txt_path}")
        _extract(zip_path, DEFAULT_RAW_NAME, raw_txt_path)
    else:
        print(f"[electricity] Using cached text file: {raw_txt_path}")

    header_cols = [str(c) for c in pd.read_csv(raw_txt_path, sep=";", decimal=",", nrows=0).columns]
    if not header_cols:
        raise ValueError(f"Could not parse header in {raw_txt_path}")
    chosen = _choose_clients(header_cols, clients, start_client, n_clients)
    first_col = header_cols[0]

    df = pd.read_csv(
        raw_txt_path,
        sep=";",
        decimal=",",
        usecols=[first_col, *chosen],
        low_memory=False,
    ).rename(columns={first_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df[chosen] = df[chosen].apply(pd.to_numeric, errors="coerce")

    if resample_frequency:
        print(f"[electricity] Resampling to {resample_frequency}")
        df = df.set_index("datetime").resample(resample_frequency).mean().reset_index()

    df = df.dropna(subset=chosen, how="all")
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    return df, chosen, zip_path, raw_txt_path


def _write_metadata(
    *, output_path: Path, url: str, zip_path: Path, raw_txt_path: Path, chosen_clients: list[str], resample_frequency: str, df: pd.DataFrame
) -> Path:
    payload = {
        "source_url": url,
        "cached_zip": str(zip_path),
        "raw_txt": str(raw_txt_path),
        "output_path": str(output_path),
        "n_rows": int(len(df)),
        "n_features": int(len(chosen_clients)),
        "selected_clients": chosen_clients,
        "resample_frequency": resample_frequency,
        "min_datetime": df["datetime"].min().isoformat() if not df.empty else None,
        "max_datetime": df["datetime"].max().isoformat() if not df.empty else None,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def prepare_electricity_csv(
    *,
    output_csv: Path,
    cache_dir: Path,
    url: str,
    force_download: bool,
    force_extract: bool,
    clients: list[str] | None,
    start_client: int,
    n_clients: int,
    resample_frequency: str,
    max_rows: int | None,
) -> tuple[Path, Path]:
    df, chosen, zip_path, raw_txt_path = _load_electricity_frame(
        cache_dir=cache_dir,
        url=url,
        force_download=force_download,
        force_extract=force_extract,
        clients=clients,
        start_client=start_client,
        n_clients=n_clients,
        resample_frequency=resample_frequency,
        max_rows=max_rows,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, ["datetime", *chosen]].to_csv(output_csv, index=False)
    meta_path = _write_metadata(
        output_path=output_csv,
        url=url,
        zip_path=zip_path,
        raw_txt_path=raw_txt_path,
        chosen_clients=chosen,
        resample_frequency=resample_frequency,
        df=df,
    )
    print(f"[electricity] Saved dataset: {output_csv}")
    print(f"[electricity] Saved metadata: {meta_path}")
    return output_csv, meta_path


def prepare_electricity_npz(
    *,
    output_npz: Path,
    cache_dir: Path,
    url: str,
    force_download: bool,
    force_extract: bool,
    clients: list[str] | None,
    start_client: int,
    n_clients: int,
    resample_frequency: str,
    max_rows: int | None,
) -> tuple[Path, Path]:
    df, chosen, zip_path, raw_txt_path = _load_electricity_frame(
        cache_dir=cache_dir,
        url=url,
        force_download=force_download,
        force_extract=force_extract,
        clients=clients,
        start_client=start_client,
        n_clients=n_clients,
        resample_frequency=resample_frequency,
        max_rows=max_rows,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    np.savez_compressed(
        output_npz,
        X=df[chosen].to_numpy(dtype=np.float32, copy=False),
        datetime=dt.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=str),
        feature_names=np.asarray(chosen, dtype=str),
    )
    meta_path = _write_metadata(
        output_path=output_npz,
        url=url,
        zip_path=zip_path,
        raw_txt_path=raw_txt_path,
        chosen_clients=chosen,
        resample_frequency=resample_frequency,
        df=df,
    )
    print(f"[electricity] Saved NPZ dataset: {output_npz}")
    print(f"[electricity] Saved metadata: {meta_path}")
    return output_npz, meta_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download and convert UCI Electricity dataset for AQI pipeline.")
    p.add_argument("--output-csv", type=Path, default=Path("data/datasets/electricity/raw/electricity.csv"))
    p.add_argument("--output-npz", type=Path, default=None)
    p.add_argument("--skip-csv", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("data/datasets/electricity/cache"))
    p.add_argument("--url", type=str, default=DEFAULT_ELECTRICITY_URL)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--clients", type=str, default=None, help="Comma-separated MT_* client names.")
    p.add_argument("--start-client", type=int, default=1, help="1-based index into available MT_* columns.")
    p.add_argument("--n-clients", type=int, default=16)
    p.add_argument("--resample-frequency", type=str, default="1h")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick smoke runs.")
    a = p.parse_args(argv)

    clients = [c.strip() for c in (a.clients or "").split(",") if c.strip()] or None
    common = dict(
        cache_dir=a.cache_dir,
        url=str(a.url),
        force_download=bool(a.force_download),
        force_extract=bool(a.force_extract),
        clients=clients,
        start_client=int(a.start_client),
        n_clients=int(a.n_clients),
        resample_frequency=str(a.resample_frequency),
        max_rows=a.max_rows,
    )
    if not bool(a.skip_csv):
        prepare_electricity_csv(output_csv=a.output_csv, **common)
    if a.output_npz is not None:
        prepare_electricity_npz(output_npz=a.output_npz, **common)


if __name__ == "__main__":
    main()
