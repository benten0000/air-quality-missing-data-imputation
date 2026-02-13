from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from air_quality_imputer.pipeline.train_models import run


class _DummyModel:
    def fit(self, *args, **kwargs):
        return {"best_loss": 0.123}

    def state_dict(self):
        return {"w": np.array([1.0], dtype=np.float32)}


class TrainModelsTests(unittest.TestCase):
    def test_train_stage_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            station = "all_stations"
            station_dir = root / "data" / "processed" / "splits" / station
            station_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                station_dir / "windows.npz",
                X_train=np.zeros((2, 4, 3), dtype=np.float32),
                X_val_masked=np.zeros((1, 4, 3), dtype=np.float32),
                X_val_ori=np.zeros((1, 4, 3), dtype=np.float32),
                X_test_ori=np.zeros((1, 4, 3), dtype=np.float32),
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "seed": 42,
                        "stations": [station],
                        "models": ["transformer"],
                        "features": ["PM10", "PM2.5", "temperature"],
                        "never_mask_features": [],
                        "block_size": 4,
                        "mask_mode": "block_feature",
                    },
                    "training": {
                        "epochs": 1,
                        "batch_size": 2,
                        "lr": 0.001,
                        "patience": 1,
                        "min_delta": 0.0,
                        "train_mask": {
                            "transformer": {
                                "mode": "random",
                                "missing_rate": 0.15,
                                "block_min_len": 2,
                                "block_max_len": 3,
                                "block_missing_prob": 0.35,
                                "feature_block_prob": 0.6,
                                "block_no_overlap": True,
                            },
                            "saits": {
                                "mode": "random",
                                "missing_rate": 0.2,
                            },
                        },
                    },
                    "paths": {
                        "processed_dir": str(root / "data" / "processed" / "splits"),
                        "models_dir": str(root / "artifacts" / "models"),
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

            with patch("air_quality_imputer.pipeline.train_models.build_model_from_cfg") as build_model:
                build_model.return_value = (_DummyModel(), {"x": 1}, "classic_transformer", "transformer.pt")
                with patch("air_quality_imputer.pipeline.train_models.config_to_dict", return_value={"x": 1}):
                    run(cfg)
                effective_model_cfg = build_model.call_args.args[0]
                self.assertEqual(str(effective_model_cfg.params.train_mask_mode), "random")
                self.assertAlmostEqual(float(effective_model_cfg.params.train_missing_rate), 0.15, places=7)

            checkpoint = root / "artifacts" / "models" / "classic_transformer" / station / "transformer.pt"
            self.assertTrue(checkpoint.exists())

    def test_train_stage_uses_manifest_features_when_config_empty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            station = "electricity"
            processed_root = root / "data" / "processed" / "splits"
            station_dir = processed_root / station
            station_dir.mkdir(parents=True, exist_ok=True)
            (processed_root / "prepare_manifest.json").write_text(
                json.dumps({"resolved_features": ["MT_001", "MT_002", "MT_003"]}),
                encoding="utf-8",
            )

            np.savez_compressed(
                station_dir / "windows.npz",
                X_train=np.zeros((2, 4, 3), dtype=np.float32),
                X_val_masked=np.zeros((1, 4, 3), dtype=np.float32),
                X_val_ori=np.zeros((1, 4, 3), dtype=np.float32),
                X_test_ori=np.zeros((1, 4, 3), dtype=np.float32),
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "seed": 42,
                        "stations": [station],
                        "models": ["transformer"],
                        "features": [],
                        "never_mask_features": [],
                        "block_size": 4,
                    },
                    "training": {
                        "epochs": 1,
                        "batch_size": 2,
                        "lr": 0.001,
                        "patience": 1,
                        "min_delta": 0.0,
                        "train_mask": {
                            "transformer": {"mode": "random", "missing_rate": 0.15},
                        },
                    },
                    "paths": {
                        "processed_dir": str(processed_root),
                        "models_dir": str(root / "artifacts" / "models"),
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

            with patch("air_quality_imputer.pipeline.train_models.build_model_from_cfg") as build_model:
                build_model.return_value = (_DummyModel(), {"x": 1}, "classic_transformer", "transformer.pt")
                with patch("air_quality_imputer.pipeline.train_models.config_to_dict", return_value={"x": 1}):
                    run(cfg)

            checkpoint = root / "artifacts" / "models" / "classic_transformer" / station / "transformer.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["features"], ["MT_001", "MT_002", "MT_003"])

    def test_saits_rejects_block_train_mask_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            station = "all_stations"
            station_dir = root / "data" / "processed" / "splits" / station
            station_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                station_dir / "windows.npz",
                X_train=np.zeros((2, 4, 3), dtype=np.float32),
                X_val_masked=np.zeros((1, 4, 3), dtype=np.float32),
                X_val_ori=np.zeros((1, 4, 3), dtype=np.float32),
                X_test_ori=np.zeros((1, 4, 3), dtype=np.float32),
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "seed": 42,
                        "stations": [station],
                        "models": ["saits"],
                        "features": ["PM10", "PM2.5", "temperature"],
                        "never_mask_features": [],
                        "block_size": 4,
                        "mask_mode": "block_feature",
                    },
                    "training": {
                        "epochs": 1,
                        "batch_size": 2,
                        "lr": 0.001,
                        "patience": 1,
                        "min_delta": 0.0,
                        "train_mask": {
                            "saits": {
                                "mode": "block",
                                "missing_rate": 0.2,
                            }
                        },
                    },
                    "paths": {
                        "processed_dir": str(root / "data" / "processed" / "splits"),
                        "models_dir": str(root / "artifacts" / "models"),
                    },
                    "tracking": {"enabled": False},
                    "models": {
                        "saits": {
                            "type": "saits",
                            "checkpoint_name": "saits.pt",
                            "params": {},
                        }
                    },
                }
            )

            with self.assertRaises(ValueError):
                run(cfg)


if __name__ == "__main__":
    unittest.main()
