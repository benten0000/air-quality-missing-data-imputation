from __future__ import annotations

import unittest
from unittest.mock import patch

from air_quality_imputer.tracking.mlflow_utils import MLflowTracker


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyMlflow:
    def __init__(self):
        self.tags = {}
        self.params = {}
        self.metrics = {}
        self.artifacts = []

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name):
        self.run_name = run_name
        return _DummyRun()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)
        self.step = step

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))

    def log_artifacts(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


class MLflowTrackerTests(unittest.TestCase):
    def test_disabled_when_mlflow_missing(self):
        with patch("air_quality_imputer.tracking.mlflow_utils._mlflow", None):
            tracker = MLflowTracker({"enabled": True})
            self.assertFalse(tracker.enabled)

    def test_logs_flattened_params_and_metrics(self):
        dummy = _DummyMlflow()
        with patch("air_quality_imputer.tracking.mlflow_utils._mlflow", dummy):
            tracker = MLflowTracker({"enabled": True, "experiment_name": "x"})
            with tracker.start_run("run-1", tags={"stage": "train"}):
                tracker.log_params({"a": {"b": 1}}, prefix="cfg")
                tracker.log_metrics({"m1": 1.5, "m2": None, "m3": "nan"})

        self.assertEqual(dummy.params["cfg.a.b"], "1")
        self.assertIn("m1", dummy.metrics)
        self.assertNotIn("m2", dummy.metrics)
        self.assertIn("stage", dummy.tags)


if __name__ == "__main__":
    unittest.main()
