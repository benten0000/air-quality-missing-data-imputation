from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ELECTRICITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
DEFAULT_RAW_NAME = "LD2011_2014.txt"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out_path.open("wb") as target:  # nosec: B310
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def _extract_zip_member(zip_path: Path, member_name: str, out_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as source:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(source.read())


def _read_header_columns(raw_txt_path: Path) -> list[str]:
    with raw_txt_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    row = next(csv.reader([first_line], delimiter=";"))
    if not row:
        raise ValueError(f"Could not parse header in {raw_txt_path}")
    return row


def _parse_clients_arg(clients: str | None) -> list[str] | None:
    if clients is None:
        return None
    chosen = [item.strip() for item in clients.split(",") if item.strip()]
    return chosen or None


def _choose_feature_columns(
    *,
    header_cols: list[str],
    clients: list[str] | None,
    start_client: int,
    n_clients: int,
) -> list[str]:
    client_cols = [col for col in header_cols if col.startswith("MT_")]
    if not client_cols:
        raise ValueError("No MT_* client columns found in Electricity file header.")
    if clients:
        missing = [name for name in clients if name not in client_cols]
        if missing:
            raise ValueError(f"Unknown Electricity clients: {missing}")
        return clients

    if start_client < 1:
        raise ValueError("--start-client must be >= 1")
    if n_clients <= 0:
        raise ValueError("--n-clients must be > 0")

    start_idx = start_client - 1
    end_idx = start_idx + n_clients
    if start_idx >= len(client_cols):
        raise ValueError(f"--start-client={start_client} is out of range (max {len(client_cols)})")
    chosen = client_cols[start_idx:end_idx]
    if not chosen:
        raise ValueError("No Electricity client columns selected.")
    return chosen


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
        _download_file(url, zip_path)
    else:
        print(f"[electricity] Using cached archive: {zip_path}")

    if force_extract or not raw_txt_path.exists():
        print(f"[electricity] Extracting {DEFAULT_RAW_NAME} to {raw_txt_path}")
        _extract_zip_member(zip_path, DEFAULT_RAW_NAME, raw_txt_path)
    else:
        print(f"[electricity] Using cached text file: {raw_txt_path}")

    header_cols = _read_header_columns(raw_txt_path)
    chosen_clients = _choose_feature_columns(
        header_cols=header_cols,
        clients=clients,
        start_client=start_client,
        n_clients=n_clients,
    )
    column_to_index = {name: idx for idx, name in enumerate(header_cols)}
    usecols = [0, *[column_to_index[name] for name in chosen_clients]]
    print(f"[electricity] Loading columns: {len(chosen_clients)} clients")
    df = pd.read_csv(
        raw_txt_path,
        sep=";",
        decimal=",",
        usecols=usecols,
        low_memory=False,
    )
    first_column = str(df.columns[0])
    df = df.rename(columns={first_column: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = _coerce_numeric(df, chosen_clients)

    if resample_frequency:
        print(f"[electricity] Resampling to {resample_frequency}")
        df = df.set_index("datetime").resample(resample_frequency).mean()
        df = df.reset_index()

    df = df.dropna(subset=chosen_clients, how="all")
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    return df, chosen_clients, zip_path, raw_txt_path


def _metadata_payload(
    *,
    output_path: Path,
    url: str,
    zip_path: Path,
    raw_txt_path: Path,
    chosen_clients: list[str],
    resample_frequency: str,
    df: pd.DataFrame,
) -> dict[str, object]:
    return {
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
    df, chosen_clients, zip_path, raw_txt_path = _load_electricity_frame(
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
    df.to_csv(output_csv, index=False)

    metadata = _metadata_payload(
        output_path=output_csv,
        url=url,
        zip_path=zip_path,
        raw_txt_path=raw_txt_path,
        chosen_clients=chosen_clients,
        resample_frequency=resample_frequency,
        df=df,
    )
    metadata_path = output_csv.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[electricity] Saved dataset: {output_csv}")
    print(f"[electricity] Saved metadata: {metadata_path}")

    return output_csv, metadata_path


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
    df, chosen_clients, zip_path, raw_txt_path = _load_electricity_frame(
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
    values = df[chosen_clients].to_numpy(dtype=np.float32, copy=False)
    datetimes = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=str)
    np.savez_compressed(
        output_npz,
        X=values,
        datetime=datetimes,
        feature_names=np.array(chosen_clients, dtype=str),
    )

    metadata = _metadata_payload(
        output_path=output_npz,
        url=url,
        zip_path=zip_path,
        raw_txt_path=raw_txt_path,
        chosen_clients=chosen_clients,
        resample_frequency=resample_frequency,
        df=df,
    )
    metadata_path = output_npz.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[electricity] Saved NPZ dataset: {output_npz}")
    print(f"[electricity] Saved metadata: {metadata_path}")

    return output_npz, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert UCI Electricity dataset for AQI pipeline.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/datasets/electricity/raw/electricity.csv"))
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/datasets/electricity/cache"))
    parser.add_argument("--url", type=str, default=DEFAULT_ELECTRICITY_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--clients",
        type=str,
        default=None,
        help="Comma-separated MT_* client names. If set, overrides --start-client/--n-clients.",
    )
    parser.add_argument("--start-client", type=int, default=1, help="1-based index into available MT_* columns.")
    parser.add_argument("--n-clients", type=int, default=16)
    parser.add_argument("--resample-frequency", type=str, default="1h")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    clients = _parse_clients_arg(args.clients)
    common_kwargs = {
        "cache_dir": args.cache_dir,
        "url": args.url,
        "force_download": bool(args.force_download),
        "force_extract": bool(args.force_extract),
        "clients": clients,
        "start_client": int(args.start_client),
        "n_clients": int(args.n_clients),
        "resample_frequency": str(args.resample_frequency),
        "max_rows": args.max_rows,
    }
    if not bool(args.skip_csv):
        prepare_electricity_csv(
            output_csv=args.output_csv,
            **common_kwargs,
        )
    if args.output_npz is not None:
        prepare_electricity_npz(
            output_npz=args.output_npz,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
