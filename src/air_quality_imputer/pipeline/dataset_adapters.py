from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from air_quality_imputer.pipeline.common import to_plain_dict


DEFAULT_DATASET_NAME = "air_quality"
COMBINED_STATION_NAME = "combined"


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
    for station in stations:
        path = data_dir / f"{station}.csv"
        if not path.exists():
            continue
        cols = [str(col) for col in pd.read_csv(path, nrows=1).columns]
        inferred = [col for col in cols if col not in {"datetime", "station"}]
        if inferred:
            return inferred
    raise ValueError(f"Could not infer CSV features in {data_dir}; set experiment.features or dataset feature_names.")


def _ensure_npz(npz_path: Path, definition: Mapping[str, Any]) -> None:
    if npz_path.exists():
        return
    ensure_cfg = definition.get("ensure")
    if not isinstance(ensure_cfg, Mapping):
        raise FileNotFoundError(f"Missing NPZ dataset: {npz_path}")
    if str(ensure_cfg.get("kind", "")).strip().lower() not in {"electricity", "electricity_uci"}:
        raise FileNotFoundError(f"Missing NPZ dataset: {npz_path} (unsupported ensure.kind).")

    from air_quality_imputer.data.electricity_dataset import DEFAULT_ELECTRICITY_URL, prepare_electricity_npz

    clients = ensure_cfg.get("clients")
    if isinstance(clients, str):
        clients = [item.strip() for item in clients.split(",") if item.strip()]
    elif isinstance(clients, list):
        clients = [str(item).strip() for item in clients if str(item).strip()]
    else:
        clients = None

    prepare_electricity_npz(
        output_npz=npz_path,
        cache_dir=Path(str(ensure_cfg.get("cache_dir", "data/datasets/electricity/cache"))),
        url=str(ensure_cfg.get("url", DEFAULT_ELECTRICITY_URL)),
        force_download=bool(ensure_cfg.get("force_download", False)),
        force_extract=bool(ensure_cfg.get("force_extract", False)),
        clients=clients,
        start_client=int(ensure_cfg.get("start_client", 1)),
        n_clients=int(ensure_cfg.get("n_clients", 16)),
        resample_frequency=str(ensure_cfg.get("resample_frequency", "1h")),
        max_rows=ensure_cfg.get("max_rows"),
    )


def _materialize_combined_wide_csv(
    *, data_dir: Path, station_names: list[str], base_features: list[str]
) -> tuple[Path, list[str]]:
    if not station_names:
        raise ValueError(f"No source stations found in {data_dir} to build combined wide station.")
    if not base_features:
        raise ValueError("No base features provided for combined wide station.")
    wide_cols = [f"{feat}_{station}" for station in station_names for feat in base_features]
    frames: list[pd.DataFrame] = []
    ref_idx: pd.DatetimeIndex | None = None
    for station in station_names:
        path = data_dir / f"{station}.csv"
        df = pd.read_csv(path, usecols=["datetime", *base_features])
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        valid = dt.notna()
        dt = dt[valid]
        values = df.loc[valid, base_features].apply(pd.to_numeric, errors="coerce")
        idx = pd.DatetimeIndex(dt)
        if ref_idx is None:
            ref_idx = idx
        elif not ref_idx.equals(idx):
            raise ValueError(f"Stations are not aligned on the same datetime index (mismatch at {path}).")
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


def _materialize_npz_to_csv(
    *, npz_path: Path, out_path: Path, definition: Mapping[str, Any], requested_features: list[str]
) -> list[str]:
    value_key = str(definition.get("value_key", "X"))
    datetime_key = str(definition.get("datetime_key", "datetime"))
    feature_names_key = str(definition.get("feature_names_key", "feature_names"))
    feature_names_cfg = definition.get("feature_names")
    feature_names_cfg = _clean_list(feature_names_cfg if isinstance(feature_names_cfg, list) else None)

    with np.load(npz_path, allow_pickle=False) as z:
        X = np.asarray(z[value_key], dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"npz[{value_key!r}] must be 2D [timesteps, features], got {X.shape}")
        n_rows, n_features = X.shape

        if feature_names_cfg:
            if len(feature_names_cfg) != n_features:
                raise ValueError("dataset.feature_names does not match npz feature width.")
            names = feature_names_cfg
        elif feature_names_key in z:
            names = [str(x) for x in np.asarray(z[feature_names_key]).reshape(-1).tolist()]
        else:
            names = [f"feature_{i+1:03d}" for i in range(n_features)]

        if datetime_key in z:
            dt = pd.to_datetime(np.asarray(z[datetime_key]).reshape(-1), errors="coerce")
        else:
            dt = pd.date_range(start="2000-01-01 00:00:00", periods=n_rows, freq="1h")
        if len(dt) != n_rows:
            raise ValueError("npz datetime length mismatch.")

    if requested_features:
        lookup = {name: i for i, name in enumerate(names)}
        missing = [name for name in requested_features if name not in lookup]
        if missing:
            raise KeyError(f"Requested features not found in dataset: {missing}")
        idx = [lookup[name] for name in requested_features]
        names = requested_features
        X = X[:, idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, columns=names).assign(datetime=dt).loc[:, ["datetime", *names]].to_csv(out_path, index=False)
    return names


def prepare_dataset_inputs(
    *,
    cfg: DictConfig,
    stations: list[str],
    requested_features: list[str],
) -> tuple[Path, dict[str, Any], list[str]]:
    dataset_name = _cfg_dataset_name(cfg)
    definition = _dataset_definition(cfg, dataset_name)
    loader = str(definition.get("loader", "air_quality_csv")).strip().lower()
    data_dir = Path(str(definition.get("data_dir", cfg.paths.data_dir)))
    requested_features = _clean_list(requested_features)

    if loader in {"air_quality_csv", "csv"}:
        features = requested_features or _clean_list(definition.get("feature_names"))
        if COMBINED_STATION_NAME in stations:
            station_names = sorted(p.stem for p in data_dir.glob("*.csv") if p.stem != COMBINED_STATION_NAME)
            base_features = _clean_list(definition.get("feature_names")) or _infer_csv_features(data_dir, station_names)
            if requested_features:
                unknown = [f for f in requested_features if f not in base_features]
                if unknown:
                    raise ValueError(f"combined expects base feature names, unknown: {unknown}")
                base_features = requested_features
            combined_path, features = _materialize_combined_wide_csv(
                data_dir=data_dir, station_names=station_names, base_features=base_features
            )
            return (
                data_dir,
                {"dataset": dataset_name, "loader": loader, "materialized_files": [str(combined_path)]},
                features,
            )

        if not features:
            features = _infer_csv_features(data_dir, stations)
        return data_dir, {"dataset": dataset_name, "loader": loader, "materialized_files": []}, features

    if loader in {"npz"}:
        if len(stations) != 1:
            raise ValueError("npz loader currently supports exactly one station.")
        npz_path_raw = definition.get("npz_path")
        if npz_path_raw is None:
            raise KeyError(f"dataset {dataset_name!r} loader=npz requires definition.npz_path")
        npz_path = Path(str(npz_path_raw))
        _ensure_npz(npz_path=npz_path, definition=definition)

        out_path = data_dir / f"{stations[0]}.csv"
        features = _materialize_npz_to_csv(
            npz_path=npz_path,
            out_path=out_path,
            definition=definition,
            requested_features=requested_features,
        )
        return (
            data_dir,
            {
                "dataset": dataset_name,
                "loader": loader,
                "npz_path": str(npz_path),
                "materialized_files": [str(out_path)],
            },
            features,
        )

    raise ValueError(f"Unsupported dataset loader {loader!r} for dataset {dataset_name!r}.")
