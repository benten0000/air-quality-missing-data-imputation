from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
                S_train=np.zeros((2,), dtype=np.int64),
                S_val=np.zeros((1,), dtype=np.int64),
                S_test=np.zeros((1,), dtype=np.int64),
                station_names=np.array(["A"], dtype=str),
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "seed": 42,
                        "stations": [station],
                        "models": ["transformer"],
                        "features": ["station", "PM10", "PM2.5"],
                        "never_mask_features": ["station"],
                        "block_size": 4,
                        "mask_mode": "block_feature",
                    },
                    "training": {"epochs": 1, "batch_size": 2, "lr": 0.001, "patience": 1, "min_delta": 0.0},
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

            checkpoint = root / "artifacts" / "models" / "classic_transformer" / station / "transformer.pt"
            self.assertTrue(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
