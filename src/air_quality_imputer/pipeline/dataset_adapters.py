from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from air_quality_imputer.pipeline.common import to_plain_dict


DEFAULT_DATASET_NAME = "air_quality"


def selected_dataset_name(cfg: DictConfig) -> str:
    exp = cfg.experiment
    name = exp.get("dataset")
    return str(name).strip() if name is not None else DEFAULT_DATASET_NAME


def resolve_dataset_definition(cfg: DictConfig, dataset_name: str) -> dict[str, Any]:
    dataset_root = cfg.get("dataset")
    if dataset_root is None:
        return {}
    if isinstance(dataset_root, DictConfig):
        definitions = dataset_root.get("definitions")
        if isinstance(definitions, DictConfig) and dataset_name in definitions:
            return to_plain_dict(definitions[dataset_name])
    plain_root = to_plain_dict(dataset_root)
    if "loader" in plain_root:
        return plain_root
    entry = plain_root.get(dataset_name)
    if isinstance(entry, Mapping):
        return {str(key): value for key, value in entry.items()}
    return {}


def _clients_from_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        clients = [item.strip() for item in value.split(",") if item.strip()]
        return clients or None
    if isinstance(value, list):
        clients = [str(item).strip() for item in value if str(item).strip()]
        return clients or None
    return None


def _clean_features(features: list[str] | None) -> list[str]:
    if not features:
        return []
    return [str(item).strip() for item in features if str(item).strip()]


def _definition_feature_names(definition: Mapping[str, Any]) -> list[str]:
    value = definition.get("feature_names")
    if not isinstance(value, list):
        return []
    return _clean_features([str(item) for item in value])


def _infer_csv_feature_names(data_dir: Path, stations: list[str]) -> list[str]:
    for station in stations:
        station_path = data_dir / f"{station}.csv"
        if not station_path.exists():
            continue
        head = pd.read_csv(station_path, nrows=1)
        cols = [str(col) for col in head.columns]
        inferred = [col for col in cols if col not in {"datetime", "station"}]
        if inferred:
            return inferred
    raise ValueError(
        f"Could not infer features from CSV files in {data_dir}. "
        "Set experiment.features or dataset.definitions.<name>.feature_names."
    )


def _ensure_npz_file(*, npz_path: Path, definition: dict[str, Any]) -> Path:
    if npz_path.exists():
        return npz_path

    ensure_cfg = definition.get("ensure")
    if not isinstance(ensure_cfg, Mapping):
        raise FileNotFoundError(
            f"NPZ dataset is missing: {npz_path}. Add dataset ensure config or provide an existing npz_path."
        )

    kind = str(ensure_cfg.get("kind", "")).strip().lower()
    if kind in {"electricity", "electricity_uci"}:
        from air_quality_imputer.data.electricity_dataset import (
            DEFAULT_ELECTRICITY_URL,
            prepare_electricity_npz,
        )

        prepare_electricity_npz(
            output_npz=npz_path,
            cache_dir=Path(str(ensure_cfg.get("cache_dir", "data/datasets/electricity/cache"))),
            url=str(ensure_cfg.get("url", DEFAULT_ELECTRICITY_URL)),
            force_download=bool(ensure_cfg.get("force_download", False)),
            force_extract=bool(ensure_cfg.get("force_extract", False)),
            clients=_clients_from_value(ensure_cfg.get("clients")),
            start_client=int(ensure_cfg.get("start_client", 1)),
            n_clients=int(ensure_cfg.get("n_clients", 16)),
            resample_frequency=str(ensure_cfg.get("resample_frequency", "1h")),
            max_rows=ensure_cfg.get("max_rows"),
        )
        return npz_path

    raise FileNotFoundError(
        f"NPZ dataset is missing: {npz_path}. Unsupported ensure kind: {kind!r}."
    )


def _default_feature_names(n_features: int) -> list[str]:
    return [f"feature_{idx + 1:03d}" for idx in range(n_features)]


def _resolve_npz_feature_names(
    *,
    payload: Mapping[str, Any],
    n_features: int,
    requested_features: list[str],
    definition: dict[str, Any],
) -> list[str]:
    if requested_features:
        if len(requested_features) != n_features:
            raise ValueError(
                f"Resolved features length ({len(requested_features)}) does not match npz feature width ({n_features})."
            )
        return requested_features

    feature_names_cfg = _definition_feature_names(definition)
    if feature_names_cfg:
        if len(feature_names_cfg) != n_features:
            raise ValueError(
                f"dataset.feature_names length ({len(feature_names_cfg)}) does not match npz feature width ({n_features})."
            )
        return feature_names_cfg

    feature_names_key = str(definition.get("feature_names_key", "feature_names"))
    if feature_names_key in payload:
        names = [str(item) for item in np.asarray(payload[feature_names_key]).tolist()]
        if len(names) != n_features:
            raise ValueError(
                f"npz key {feature_names_key!r} has {len(names)} names, expected {n_features}."
            )
        return names

    return _default_feature_names(n_features)


def _resolve_npz_datetime(
    *,
    payload: Mapping[str, Any],
    n_rows: int,
    definition: dict[str, Any],
) -> pd.DatetimeIndex:
    datetime_key = str(definition.get("datetime_key", "datetime"))
    if datetime_key in payload:
        dt = pd.to_datetime(np.asarray(payload[datetime_key]).reshape(-1), errors="coerce")
        if len(dt) != n_rows:
            raise ValueError(f"npz key {datetime_key!r} has {len(dt)} rows, expected {n_rows}.")
        if dt.isna().all():
            raise ValueError(f"npz key {datetime_key!r} does not contain parseable timestamps.")
        return pd.DatetimeIndex(dt)

    start_datetime = str(definition.get("start_datetime", "2000-01-01 00:00:00"))
    frequency = str(definition.get("frequency", "1h"))
    return pd.date_range(start=start_datetime, periods=n_rows, freq=frequency)


def _station_to_mask(
    *,
    station: str,
    station_ids: np.ndarray | None,
    station_name_to_id: dict[str, int],
    n_rows: int,
) -> np.ndarray:
    if station_ids is None:
        return np.ones((n_rows,), dtype=bool)

    if station in station_name_to_id:
        sid = int(station_name_to_id[station])
        return station_ids == sid

    try:
        sid = int(station)
    except Exception as exc:
        expected = sorted(station_name_to_id)
        raise KeyError(f"Station {station!r} not found in station_names; expected one of {expected}") from exc
    return station_ids == sid


def materialize_npz_timeseries_to_csv(
    *,
    npz_path: Path,
    data_dir: Path,
    stations: list[str],
    requested_features: list[str],
    definition: dict[str, Any],
) -> tuple[list[Path], list[str]]:
    value_key = str(definition.get("value_key", "X"))
    station_ids_key = str(definition.get("station_ids_key", "station_ids"))
    station_names_key = str(definition.get("station_names_key", "station_names"))
    max_rows_raw = definition.get("max_rows")
    max_rows = None if max_rows_raw is None else int(max_rows_raw)

    with np.load(npz_path, allow_pickle=False) as payload:
        if value_key not in payload:
            raise KeyError(f"Missing key {value_key!r} in npz: {npz_path}")
        values = np.asarray(payload[value_key], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                f"npz key {value_key!r} must be a 2D array [timesteps, features], got shape {values.shape}"
            )
        n_rows, n_features = values.shape
        feature_names = _resolve_npz_feature_names(
            payload=payload,
            n_features=n_features,
            requested_features=requested_features,
            definition=definition,
        )
        datetimes = _resolve_npz_datetime(payload=payload, n_rows=n_rows, definition=definition)

        station_ids: np.ndarray | None = None
        station_name_to_id: dict[str, int] = {}
        if station_ids_key in payload:
            station_ids = np.asarray(payload[station_ids_key]).reshape(-1)
            if len(station_ids) != n_rows:
                raise ValueError(
                    f"npz key {station_ids_key!r} has {len(station_ids)} rows, expected {n_rows}."
                )
            if station_names_key in payload:
                station_names = [str(item) for item in np.asarray(payload[station_names_key]).tolist()]
                station_name_to_id = {name: idx for idx, name in enumerate(station_names)}

        written_paths: list[Path] = []
        data_dir.mkdir(parents=True, exist_ok=True)

        for station in stations:
            mask = _station_to_mask(
                station=station,
                station_ids=station_ids,
                station_name_to_id=station_name_to_id,
                n_rows=n_rows,
            )
            station_values = values[mask]
            station_datetimes = datetimes[mask]
            if max_rows is not None and max_rows > 0 and len(station_values) > max_rows:
                station_values = station_values[:max_rows]
                station_datetimes = station_datetimes[:max_rows]
            if len(station_values) == 0:
                raise ValueError(f"No rows found for station {station!r} in npz dataset {npz_path}")

            station_df = pd.DataFrame(station_values, columns=feature_names)
            station_df = station_df.assign(datetime=station_datetimes, station=station)
            station_df = station_df[["datetime", "station", *feature_names]]
            station_path = data_dir / f"{station}.csv"
            station_df.to_csv(station_path, index=False)
            written_paths.append(station_path)

    return written_paths, feature_names


def prepare_dataset_inputs(
    *,
    cfg: DictConfig,
    stations: list[str],
    requested_features: list[str],
) -> tuple[Path, dict[str, Any], list[str]]:
    dataset_name = selected_dataset_name(cfg)
    definition = resolve_dataset_definition(cfg, dataset_name)
    loader = str(definition.get("loader", "air_quality_csv")).strip().lower()
    data_dir = Path(str(definition.get("data_dir", cfg.paths.data_dir)))
    requested_features = _clean_features(requested_features)

    if loader in {"air_quality_csv", "csv_station_files", "csv"}:
        resolved_features = requested_features or _definition_feature_names(definition)
        if not resolved_features:
            resolved_features = _infer_csv_feature_names(data_dir, stations)
        return data_dir, {"dataset": dataset_name, "loader": loader, "materialized_files": []}, resolved_features

    if loader in {"npz", "npz_timeseries"}:
        npz_path_raw = definition.get("npz_path")
        if npz_path_raw is None:
            raise KeyError(f"dataset {dataset_name!r} with loader=npz requires definition.npz_path")
        npz_path = Path(str(npz_path_raw))
        npz_path = _ensure_npz_file(npz_path=npz_path, definition=definition)
        written_paths, resolved_features = materialize_npz_timeseries_to_csv(
            npz_path=npz_path,
            data_dir=data_dir,
            stations=stations,
            requested_features=requested_features,
            definition=definition,
        )
        return (
            data_dir,
            {
                "dataset": dataset_name,
                "loader": loader,
                "npz_path": str(npz_path),
                "materialized_files": [str(path) for path in written_paths],
            },
            resolved_features,
        )

    raise ValueError(
        f"Unsupported dataset loader {loader!r} for dataset {dataset_name!r}. "
        "Supported: air_quality_csv, npz."
    )
