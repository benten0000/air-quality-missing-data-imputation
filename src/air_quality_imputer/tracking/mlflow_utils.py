import contextlib
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping

from air_quality_imputer import exceptions, logger


try:
    import mlflow as _mlflow
except Exception:  # pragma: no cover
    _mlflow = None

_dagshub: Any | None
try:
    import dagshub as _dagshub
except Exception:  # pragma: no cover
    _dagshub = None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in values.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(_flatten(v, key)) if isinstance(v, Mapping) else out.setdefault(key, v)
    return out


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower()).strip("-")
    return "-".join(part for part in cleaned.split("-") if part) or "dataset"


def _experiment_name(dataset_name: str) -> str:
    slug = _slug(dataset_name)
    return "air-quality-imputer" if slug in {"air", "air-quality", "airquality"} else f"{slug}-quality-imputer"


class MLflowTracker:
    def __init__(self, tracking_cfg: Mapping[str, Any] | None):
        cfg = dict(tracking_cfg or {})
        self.enabled = bool(cfg.get("enabled", True)) and _mlflow is not None
        self._mlflow: Any | None = _mlflow
        self.dataset_name = str(cfg.get("dataset_name", "air_quality")).strip() or "air_quality"
        self.base_tags = {"git_commit": _git_commit(), "project": "air-quality-imputer"}
        if not self.enabled:
            if bool(cfg.get("enabled", True)) and _mlflow is None:
                logger.logger.warning("[WARN] MLflow is not installed, tracking is disabled.")
            return

        repo_owner, repo_name = cfg.get("repo_owner"), cfg.get("repo_name")
        if repo_owner and repo_name and _dagshub is not None:
            try:
                init_fn = getattr(_dagshub, "init", None)
                if callable(init_fn):
                    init_fn(repo_owner=str(repo_owner), repo_name=str(repo_name), mlflow=True)
                else:
                    logger.logger.warning("[WARN] dagshub.init is not available, skipping DagsHub integration.")
            except Exception as exc:
                logger.logger.warning(f"[WARN] dagshub.init failed: {exc}")

        try:
            mlflow_inst = self._mlflow
            if mlflow_inst is None:
                raise exceptions.TrackingError("MLflow module is not available.")
            mlflow_inst.set_experiment(_experiment_name(self.dataset_name))
        except Exception as exc:
            logger.logger.warning(f"[WARN] MLflow init failed, tracking is disabled: {exc}")
            self.enabled = False

    @contextlib.contextmanager
    def start_run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled or self._mlflow is None:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name) as run:
            tag_map = {**self.base_tags, **dict(tags or {})}
            tag_map.setdefault("dataset", self.dataset_name)
            tag_map.setdefault("Dataset", self.dataset_name)
            if "model" in tag_map and "models" not in tag_map:
                tag_map["models"] = tag_map["model"]
            if "models" in tag_map and "Models" not in tag_map:
                tag_map["Models"] = tag_map["models"]
            self._mlflow.set_tags({k: str(v) for k, v in tag_map.items() if v is not None})

            # Keep Dataset/Models visible as table columns in UIs that expose params.
            params_map: dict[str, Any] = {"dataset": self.dataset_name, "Dataset": self.dataset_name}
            if "models" in tag_map:
                params_map["models"] = tag_map["models"]
                params_map["Models"] = tag_map["models"]
            self._mlflow.log_params({k: str(v) for k, v in params_map.items() if v is not None})

            # Populate MLflow Run Inputs -> Dataset column in some UIs (DagsHub/MLflow).
            try:
                import numpy as np
                import mlflow.data as mlflow_data

                ds = mlflow_data.from_numpy(
                    np.zeros((1, 1), dtype=np.float32),
                    source=str(self.dataset_name),
                    name=str(self.dataset_name),
                )
                self._mlflow.log_input(dataset=ds, context="pipeline")
            except Exception:
                pass
            yield run

    def log_params(self, params: Mapping[str, Any], prefix: str | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        flat = _flatten(params)
        if prefix:
            flat = {f"{prefix}.{k}": v for k, v in flat.items()}
        payload = {k: str(v) for k, v in flat.items() if v is not None}
        if payload:
            self._mlflow.log_params(payload)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        payload: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                n = float(v)
            except Exception:
                continue
            if n == n:
                payload[str(k)] = n
        if payload:
            self._mlflow.log_metrics(payload) if step is None else self._mlflow.log_metrics(payload, step=step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        if self.enabled and self._mlflow is not None and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_model_stub(
        self,
        *,
        local_path: Path,
        model_name: str = "model",
    ) -> bool:
        """Log a tiny MLflow model artifact so UI "Models" is populated."""
        if not self.enabled or self._mlflow is None:
            return False
        if not local_path.exists():
            return False
        try:
            import mlflow.pyfunc

            class _CheckpointStubModel(mlflow.pyfunc.PythonModel):
                def predict(self, context, model_input):  # pragma: no cover
                    return model_input

            mlflow.pyfunc.log_model(
                name=str(model_name),
                python_model=_CheckpointStubModel(),
                artifacts={"checkpoint": str(local_path)},
            )
            return True
        except Exception as exc:
            logger.logger.warning(f"[WARN] Failed to log MLflow model stub: {exc}")
            return False

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
