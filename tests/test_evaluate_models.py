from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from air_quality_imputer.pipeline.evaluate_models import run


class EvaluateModelsTests(unittest.TestCase):
    def test_eval_stage_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "stations": ["all_stations"],
                        "models": ["transformer"],
                        "missing_rate": 0.2,
                        "seed": 42,
                        "mask_mode": "block_feature",
                        "block_min_len": 2,
                        "block_max_len": 3,
                        "block_missing_prob": 0.35,
                        "feature_block_prob": 0.6,
                        "block_no_overlap": True,
                        "never_mask_features": ["station"],
                    },
                    "paths": {
                        "models_dir": str(root / "artifacts" / "models"),
                        "processed_dir": str(root / "data" / "processed" / "splits"),
                        "reports_dir": str(root / "reports" / "model_eval"),
                        "plots_dir": str(root / "reports" / "plots" / "model_eval"),
                        "metrics_dir": str(root / "reports" / "metrics"),
                    },
                    "tracking": {"enabled": False},
                    "models": {
                        "transformer": {
                            "type": "classic_transformer",
                            "checkpoint_name": "transformer.pt",
                            "params": {},
                        }
                    },
                }
            )

            with patch("air_quality_imputer.pipeline.evaluate_models.evaluate_station") as eval_station:
                eval_station.return_value = (
                    {"station": "all_stations", "n_eval": 10, "mae": 0.1, "rmse": 0.2},
                    [
                        {
                            "station": "all_stations",
                            "feature": "PM10",
                            "n_eval": 10,
                            "mae": 0.1,
                            "rmse": 0.2,
                        }
                    ],
                )
                with patch("air_quality_imputer.pipeline.evaluate_models.save_eval_plots", return_value=[]):
                    run(cfg)

            metrics_path = root / "reports" / "metrics" / "model_eval_metrics.json"
            self.assertTrue(metrics_path.exists())
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("models", payload)
            self.assertIn("classic_transformer", payload["models"])
            model_metrics = payload["models"]["classic_transformer"]
            self.assertIn("mae", model_metrics)
            self.assertIn("rmse", model_metrics)
            self.assertIn("n_eval", model_metrics)


if __name__ == "__main__":
    unittest.main()
