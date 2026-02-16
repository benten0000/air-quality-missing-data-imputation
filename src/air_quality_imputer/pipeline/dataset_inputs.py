from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd
from omegaconf import DictConfig

from air_quality_imputer import exceptions
from air_quality_imputer.pipeline.common import to_plain_dict


DEFAULT_DATASET_NAME = "air_quality"
COMBINED_STATION_NAME = "combined"
CSV_LOADERS = {"air_quality_csv", "csv", "physionet_csv", "ett_csv", "electricity_csv"}


def _cfg_dataset_name(cfg: DictConfig) -> str:
    value = cfg.experiment.get("dataset")
    name = str(value).strip() if value is not None else ""
    return name or DEFAULT_DATASET_NAME


def _dataset_definition(cfg: DictConfig, dataset_name: str) -> dict[str, Any]:
    root = cfg.get("dataset")
    if root is None:
        return {}
    if isinstance(root, DictConfig):
        defs = root.get("definitions")
        if isinstance(defs, DictConfig) and dataset_name in defs:
            return to_plain_dict(defs[dataset_name])
    return {}


def _clean_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _infer_csv_features(data_dir: Path, stations: list[str]) -> list[str]:
    candidates: list[Path] = []
    for station in stations:
        candidates.append(data_dir / f"{station}.csv")
    for path in sorted(data_dir.glob("*.csv")):
        if path.stem != COMBINED_STATION_NAME:
            candidates.append(path)

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        cols = [str(col) for col in pd.read_csv(path, nrows=1).columns]
        inferred = [col for col in cols if col not in {"datetime", "station", "series_id"}]
        if inferred:
            return inferred
    raise exceptions.ValidationError(f"Could not infer CSV features in {data_dir}.")


def _materialize_combined_wide_csv(
    *,
    data_dir: Path,
    station_names: list[str],
    base_features: list[str],
) -> tuple[Path, list[str]]:
    if not station_names:
        raise exceptions.ValidationError(f"No source stations found in {data_dir} to build combined station.")
    if not base_features:
        raise exceptions.ValidationError("No base features provided for combined station.")

    wide_cols = [f"{feat}_{station}" for station in station_names for feat in base_features]
    frames: list[pd.DataFrame] = []
    ref_idx: pd.DatetimeIndex | None = None
    for station in station_names:
        path = data_dir / f"{station}.csv"
        df = pd.read_csv(path, usecols=["datetime", *base_features])
        dt = cast(pd.Series, pd.to_datetime(df["datetime"], errors="coerce"))
        valid = dt.notna()
        dt = dt.loc[valid]
        values = df.loc[valid, base_features].apply(pd.to_numeric, errors="coerce")
        idx = pd.DatetimeIndex(dt)
        if ref_idx is None:
            ref_idx = idx
        elif not ref_idx.equals(idx):
            raise exceptions.ValidationError(f"Stations are not aligned on datetime index ({path}).")
        frame = values.set_index(idx)
        frame = frame.rename(columns={feat: f"{feat}_{station}" for feat in base_features})
        frames.append(frame)

    combined = pd.concat(frames, axis=1, join="inner").sort_index()
    combined = combined.reset_index(names="datetime")
    combined = combined[["datetime", *wide_cols]]
    out_path = data_dir / f"{COMBINED_STATION_NAME}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return out_path, wide_cols


def prepare_dataset_inputs(
    *,
    cfg: DictConfig,
    stations: list[str],
    requested_features: list[str],
) -> tuple[Path, dict[str, Any], list[str]]:
    dataset_name = _cfg_dataset_name(cfg)
    definition = _dataset_definition(cfg, dataset_name)
    loader = str(definition.get("loader", "csv")).strip().lower()
    if loader not in CSV_LOADERS:
        raise exceptions.ValidationError(
            f"Dataset {dataset_name!r} uses loader={loader!r}. This pipeline supports CSV-only datasets."
        )

    data_dir = Path(str(definition.get("data_dir", cfg.paths.data_dir)))
    requested_features = _clean_list(requested_features)

    split_cfg = definition.get("split")
    split_payload = to_plain_dict(split_cfg) if isinstance(split_cfg, Mapping) else {}
    window_cfg = definition.get("window")
    window_payload = to_plain_dict(window_cfg) if isinstance(window_cfg, Mapping) else {}

    if COMBINED_STATION_NAME in stations:
        combined_path = data_dir / f"{COMBINED_STATION_NAME}.csv"
        materialized_files: list[str] = []
        if not combined_path.exists():
            station_names = sorted(p.stem for p in data_dir.glob("*.csv") if p.stem != COMBINED_STATION_NAME)
            base_features = requested_features or _infer_csv_features(data_dir, station_names)
            if requested_features:
                inferred = _infer_csv_features(data_dir, station_names)
                unknown = [f for f in requested_features if f not in inferred]
                if unknown:
                    raise exceptions.ValidationError(f"combined expects base feature names, unknown: {unknown}")
            combined_path, features = _materialize_combined_wide_csv(
                data_dir=data_dir,
                station_names=station_names,
                base_features=base_features,
            )
            materialized_files = [str(combined_path)]
        else:
            features = requested_features or _infer_csv_features(data_dir, [COMBINED_STATION_NAME])

        return (
            data_dir,
            {
                "dataset": dataset_name,
                "loader": loader,
                "source_format": "csv",
                "materialized_files": materialized_files,
                "split": split_payload,
                "window": window_payload,
            },
            features,
        )

    features = requested_features
    if not features:
        features = _infer_csv_features(data_dir, stations)
    return (
        data_dir,
        {
            "dataset": dataset_name,
            "loader": loader,
            "source_format": "csv",
            "materialized_files": [],
            "split": split_payload,
            "window": window_payload,
        },
        features,
    )
