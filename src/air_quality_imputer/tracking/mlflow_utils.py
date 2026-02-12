from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, cast


try:
    import mlflow as _mlflow
except Exception:  # pragma: no cover - exercised in tests with monkeypatch
    _mlflow = None

try:
    import dagshub as _dagshub
except Exception:  # pragma: no cover - optional dependency
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
            cleaned[key] = number
    return cleaned


def _slug(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


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
        if _dagshub is None:
            print("[WARN] dagshub package is not installed, tracking is disabled.")
            self.enabled = False
            return

        mlflow = self._client()
        self._init_dagshub(cfg)
        try:
            mlflow.set_experiment(str(cfg.get("experiment_name", "air-quality-imputer")))
        except Exception as exc:
            print(f"[WARN] MLflow init failed, disabling tracking: {exc}")
            self.enabled = False

    def _init_dagshub(self, cfg: Mapping[str, Any]) -> None:
        if _dagshub is None:
            print("[WARN] dagshub package is not installed, skipping dagshub.init.")
            return
        repo_owner = cfg.get("repo_owner") or os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = cfg.get("repo_name") or os.getenv("DAGSHUB_REPO_NAME")
        if not repo_owner or not repo_name:
            raise RuntimeError("Missing repo_owner/repo_name for dagshub.init.")
        try:
            _dagshub.init(  # pyright: ignore[reportPrivateImportUsage]
                repo_owner=str(repo_owner),
                repo_name=str(repo_name),
                mlflow=True,
            )
        except Exception as exc:
            raise RuntimeError(f"dagshub.init failed: {exc}") from exc

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

    def log_torch_model(
        self,
        model: Any,
        artifact_path: str,
        model_name: str | None = None,
        registered_model_name: str | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        try:
            import torch.nn as nn
            if not isinstance(model, nn.Module):
                return False
            mlflow_pytorch = importlib.import_module("mlflow.pytorch")
            kwargs: dict[str, Any] = {}
            if model_name:
                kwargs["name"] = str(model_name)
            if registered_model_name:
                kwargs["registered_model_name"] = str(registered_model_name)
            mlflow_pytorch.log_model(model, artifact_path=artifact_path, **kwargs)
            return True
        except Exception as exc:
            print(f"[WARN] Failed to log MLflow torch model: {exc}")
            return False

    def log_checkpoint_pyfunc_model(
        self,
        checkpoint_path: Path | str,
        artifact_path: str,
        model_name: str | None = None,
        registered_model_name: str | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            return False
        try:
            pyfunc = importlib.import_module("mlflow.pyfunc")

            class _CheckpointPyfunc(pyfunc.PythonModel):  # type: ignore[misc,valid-type]
                def load_context(self, context):  # type: ignore[no-untyped-def]
                    import torch
                    from air_quality_imputer.training.model_registry import build_model_from_checkpoint

                    payload = torch.load(context.artifacts["checkpoint"], map_location=torch.device("cpu"), weights_only=False)
                    self._model = build_model_from_checkpoint(str(payload["model_type"]), payload["config_dict"])
                    self._model.load_state_dict(payload["state_dict"])
                    self._model.to(torch.device("cpu"))
                    self._model.eval()

                def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
                    del context, params
                    import numpy as np
                    import pandas as pd

                    x: Any
                    if isinstance(model_input, pd.DataFrame):
                        x = model_input.to_numpy(dtype=np.float32)
                    else:
                        x = np.asarray(model_input, dtype=np.float32)
                    if x.ndim == 2:
                        x = x[None, :, :]
                    if x.ndim != 3:
                        raise ValueError(f"Expected [batch, steps, features], got shape {x.shape}")
                    y = self._model.impute({"X": x})
                    if y is None:
                        raise RuntimeError("Model impute returned None")
                    return y

            kwargs: dict[str, Any] = {}
            if model_name:
                kwargs["name"] = str(model_name)
            if registered_model_name:
                kwargs["registered_model_name"] = str(registered_model_name)
            pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_CheckpointPyfunc(),
                artifacts={"checkpoint": str(ckpt)},
                **kwargs,
            )
            return True
        except Exception as exc:
            print(f"[WARN] Failed to log MLflow pyfunc model from checkpoint: {exc}")
            return False

    def build_registered_model_name(self, model_name: str, station: str) -> str:
        return f"aqi-{_slug(model_name)}-{_slug(station)}"
