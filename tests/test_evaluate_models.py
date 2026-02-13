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
                        "seed": 42,
                        "never_mask_features": ["station"],
                    },
                    "evaluation": {
                        "missing_rate": 0.2,
                        "mask_mode": "block_feature",
                        "block_min_len": 2,
                        "block_max_len": 3,
                        "block_missing_prob": 0.35,
                        "feature_block_prob": 0.6,
                        "block_no_overlap": True,
                        "model_ids": {},
                        "mlflow_model_refs": {},
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

    def test_eval_stage_uses_selected_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_path = root / "artifacts" / "models" / "classic_transformer" / "all_stations" / "transformer.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"x")
            model_index_path = root / "artifacts" / "models" / "model_index.json"
            model_index_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "model_id": "transformer-picked",
                                "model_name": "transformer",
                                "model_type": "classic_transformer",
                                "station": "all_stations",
                                "checkpoint_path": str(model_path),
                                "train_run_name": "train/transformer/all_stations/seed-1",
                                "train_run_id": "abc123",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "stations": ["all_stations"],
                        "models": ["transformer"],
                        "seed": 42,
                        "never_mask_features": [],
                    },
                    "evaluation": {
                        "missing_rate": 0.2,
                        "mask_mode": "block_feature",
                        "block_min_len": 2,
                        "block_max_len": 3,
                        "block_missing_prob": 0.35,
                        "feature_block_prob": 0.6,
                        "block_no_overlap": True,
                        "model_ids": {"transformer": "transformer-picked"},
                        "mlflow_model_refs": {},
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

            self.assertEqual(eval_station.call_args.kwargs["model_path"], model_path)

    def test_eval_stage_prefers_mlflow_ref_over_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_path_by_id = root / "artifacts" / "models" / "classic_transformer" / "all_stations" / "by_id.pt"
            model_path_by_id.parent.mkdir(parents=True, exist_ok=True)
            model_path_by_id.write_bytes(b"id")
            model_path_mlflow = root / "downloads" / "from_mlflow.pt"
            model_path_mlflow.parent.mkdir(parents=True, exist_ok=True)
            model_path_mlflow.write_bytes(b"mlflow")
            model_index_path = root / "artifacts" / "models" / "model_index.json"
            model_index_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "model_id": "transformer-picked",
                                "model_name": "transformer",
                                "model_type": "classic_transformer",
                                "station": "all_stations",
                                "checkpoint_path": str(model_path_by_id),
                                "train_run_name": "train/transformer/all_stations/seed-1",
                                "train_run_id": "abc123",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "stations": ["all_stations"],
                        "models": ["transformer"],
                        "seed": 42,
                        "never_mask_features": [],
                    },
                    "evaluation": {
                        "missing_rate": 0.2,
                        "mask_mode": "block_feature",
                        "block_min_len": 2,
                        "block_max_len": 3,
                        "block_missing_prob": 0.35,
                        "feature_block_prob": 0.6,
                        "block_no_overlap": True,
                        "model_ids": {"transformer": "transformer-picked"},
                        "mlflow_model_refs": {"transformer": "runs:/dummy/model/files/transformer.pt"},
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
                with patch(
                    "air_quality_imputer.pipeline.evaluate_models._resolve_checkpoint_from_mlflow_ref",
                    return_value=model_path_mlflow,
                ):
                    with patch("air_quality_imputer.pipeline.evaluate_models.save_eval_plots", return_value=[]):
                        run(cfg)

            self.assertEqual(eval_station.call_args.kwargs["model_path"], model_path_mlflow)


if __name__ == "__main__":
    unittest.main()
