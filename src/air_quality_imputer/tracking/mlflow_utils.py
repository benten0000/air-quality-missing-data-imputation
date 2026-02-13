from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping


try:
    import mlflow as _mlflow
except Exception:  # pragma: no cover
    _mlflow = None

try:
    import dagshub as _dagshub
except Exception:  # pragma: no cover
    _dagshub = None


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
            cleaned[str(key)] = number
    return cleaned


def _slug(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "dataset"


def _dataset_experiment_name(dataset_name: str) -> str:
    dataset_slug = _slug(dataset_name)
    if dataset_slug in {"air", "air-quality", "airquality"}:
        return "air-quality-imputer"
    return f"{dataset_slug}-quality-imputer"


class MLflowTracker:
    """Small wrapper over MLflow with safe no-op fallback."""

    def __init__(self, tracking_cfg: Mapping[str, Any] | None):
        cfg = dict(tracking_cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self._mlflow: Any | None = _mlflow
        self.base_tags = {"git_commit": _git_commit(), "project": "air-quality-imputer"}

        if not self.enabled:
            return
        if self._mlflow is None:
            print("[WARN] MLflow is not installed, tracking is disabled.")
            self.enabled = False
            return

        repo_owner = cfg.get("repo_owner")
        repo_name = cfg.get("repo_name")
        if repo_owner and repo_name and _dagshub is not None:
            try:
                _dagshub.init(repo_owner=str(repo_owner), repo_name=str(repo_name), mlflow=True)
            except Exception as exc:
                print(f"[WARN] dagshub.init failed: {exc}")

        dataset_name = str(cfg.get("dataset_name", "air_quality")).strip() or "air_quality"
        experiment_name = _dataset_experiment_name(dataset_name)
        try:
            self._client().set_experiment(experiment_name)
        except Exception as exc:
            print(f"[WARN] MLflow init failed, tracking is disabled: {exc}")
            self.enabled = False

    def _client(self) -> Any:
        if self._mlflow is None:
            raise RuntimeError("MLflow client is unavailable.")
        return self._mlflow

    @contextlib.contextmanager
    def start_run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        mlflow = self._client()
        with mlflow.start_run(run_name=run_name) as run:
            run_tags = {k: str(v) for k, v in {**self.base_tags, **dict(tags or {})}.items()}
            mlflow.set_tags(run_tags)
            yield run

    def log_params(self, params: Mapping[str, Any], prefix: str | None = None) -> None:
        if not self.enabled:
            return
        flat = _flatten_dict(params)
        if prefix:
            flat = {f"{prefix}.{key}": value for key, value in flat.items()}
        payload = {k: str(v) for k, v in flat.items() if v is not None}
        if payload:
            self._client().log_params(payload)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        payload = _clean_metrics(metrics)
        if not payload:
            return
        mlflow = self._client()
        mlflow.log_metrics(payload) if step is None else mlflow.log_metrics(payload, step=step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._client().log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self.enabled:
            self._client().set_tags({k: str(v) for k, v in tags.items()})

