from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, cast


try:
    import mlflow as _mlflow
except Exception:  # pragma: no cover - exercised in tests with monkeypatch
    _mlflow = None


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _flatten_dict(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def _clean_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except Exception:
            continue
        if number == number:
            cleaned[key] = number
    return cleaned


class MLflowTracker:
    """Small wrapper over MLflow with safe no-op fallback."""

    def __init__(self, tracking_cfg: Mapping[str, Any] | None):
        cfg = dict(tracking_cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self._mlflow: Any | None = _mlflow
        self.base_tags = {"git_commit": _git_commit(), "project": "air-quality-imputer"}
        if not self.enabled:
            return
        if self.enabled and not self._mlflow:
            print("[WARN] MLflow is not installed, tracking is disabled.")
            self.enabled = False
            return

        mlflow = self._client()
        tracking_uri = cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(cfg.get("experiment_name", "air-quality-imputer")))

    def _client(self) -> Any:
        if self._mlflow is None:
            raise RuntimeError("MLflow client is unavailable.")
        return cast(Any, self._mlflow)

    @contextlib.contextmanager
    def start_run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        mlflow = self._client()
        run_tags = {k: str(v) for k, v in {**self.base_tags, **dict(tags or {})}.items()}
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(run_tags)
            yield run

    def log_params(self, params: Mapping[str, Any], prefix: str | None = None) -> None:
        if not self.enabled:
            return
        mlflow = self._client()
        flat = _flatten_dict(params)
        if prefix:
            flat = {f"{prefix}.{key}": value for key, value in flat.items()}
        payload = {k: str(v) for k, v in flat.items() if v is not None}
        if payload:
            mlflow.log_params(payload)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        mlflow = self._client()
        payload = _clean_metrics(metrics)
        if not payload:
            return
        mlflow.log_metrics(payload) if step is None else mlflow.log_metrics(payload, step=step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._client().log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: Path | str, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._client().log_artifacts(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self.enabled:
            self._client().set_tags({k: str(v) for k, v in tags.items()})
