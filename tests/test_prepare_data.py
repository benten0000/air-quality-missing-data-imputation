from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from air_quality_imputer.pipeline.prepare_data import run


class PrepareDataTests(unittest.TestCase):
    def test_prepare_stage_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)

            rows = 40
            df = pd.DataFrame(
                {
                    "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
                    "station": ["A"] * rows,
                    "PM10": np.linspace(1, 2, rows),
                    "PM2.5": np.linspace(2, 3, rows),
                    "temperature": np.linspace(3, 4, rows),
                    "rain": np.linspace(4, 5, rows),
                    "pressure": np.linspace(5, 6, rows),
                    "precipitation": np.linspace(6, 7, rows),
                    "wind_speed": np.linspace(7, 8, rows),
                    "clouds": np.linspace(8, 9, rows),
                    "wind_direction": np.linspace(9, 10, rows),
                }
            )
            df.to_csv(raw_dir / "all_stations.csv", index=False)

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "stations": ["all_stations"],
                        "features": [
                            "station",
                            "PM10",
                            "PM2.5",
                            "temperature",
                            "rain",
                            "pressure",
                            "precipitation",
                            "wind_speed",
                            "clouds",
                            "wind_direction",
                        ],
                        "never_mask_features": ["station"],
                        "block_size": 4,
                        "step_size": 1,
                        "seed": 42,
                    },
                    "training": {
                        "shared_validation_mask": {
                            "missing_rate": 0.2,
                            "mask_mode": "block_feature",
                            "block_min_len": 2,
                            "block_max_len": 3,
                            "block_missing_prob": 0.35,
                            "feature_block_prob": 0.6,
                            "block_no_overlap": True,
                        }
                    },
                    "paths": {
                        "data_dir": str(raw_dir),
                        "processed_dir": str(root / "data" / "processed" / "splits"),
                        "scalers_dir": str(root / "data" / "processed" / "scalers"),
                    },
                }
            )

            run(cfg)

            split_root = root / "data" / "processed" / "splits" / "all_stations"
            scaler_path = root / "data" / "processed" / "scalers" / "all_stations" / "scaler.pkl"
            self.assertTrue((split_root / "windows.npz").exists())
            self.assertTrue((split_root / "train_data.csv").exists())
            self.assertTrue((split_root / "val_data.csv").exists())
            self.assertTrue((split_root / "test_data.csv").exists())
            self.assertTrue(scaler_path.exists())
            self.assertTrue((root / "data" / "processed" / "splits" / "prepare_manifest.json").exists())

    def test_prepare_stage_materializes_npz_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "datasets" / "toy_electricity.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)

            rows = 72
            features = ["MT_001", "MT_002", "MT_003"]
            values = np.stack(
                [
                    np.linspace(1, 3, rows),
                    np.linspace(10, 20, rows),
                    np.linspace(100, 200, rows),
                ],
                axis=1,
            ).astype(np.float32)
            datetimes = (
                pd.date_range("2024-01-01", periods=rows, freq="h")
                .strftime("%Y-%m-%d %H:%M:%S")
                .to_numpy(dtype=str)
            )
            np.savez_compressed(
                npz_path,
                X=values,
                datetime=datetimes,
                feature_names=np.array(features, dtype=str),
            )

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "dataset": "electricity",
                        "stations": ["electricity"],
                        "features": [],
                        "never_mask_features": [],
                        "block_size": 12,
                        "step_size": 1,
                        "seed": 42,
                    },
                    "dataset": {
                        "definitions": {
                            "electricity": {
                                "loader": "npz",
                                "npz_path": str(npz_path),
                                "data_dir": str(root / "data" / "materialized"),
                                "feature_names": features,
                                "value_key": "X",
                                "datetime_key": "datetime",
                                "feature_names_key": "feature_names",
                            }
                        }
                    },
                    "training": {
                        "shared_validation_mask": {
                            "missing_rate": 0.2,
                            "mask_mode": "random",
                            "block_min_len": 2,
                            "block_max_len": 3,
                            "block_missing_prob": 0.35,
                            "feature_block_prob": 0.6,
                            "block_no_overlap": True,
                        }
                    },
                    "paths": {
                        "data_dir": str(root / "data" / "unused"),
                        "processed_dir": str(root / "data" / "processed" / "splits"),
                        "scalers_dir": str(root / "data" / "processed" / "scalers"),
                    },
                }
            )

            run(cfg)

            self.assertTrue((root / "data" / "materialized" / "electricity.csv").exists())
            split_root = root / "data" / "processed" / "splits" / "electricity"
            self.assertTrue((split_root / "windows.npz").exists())
            manifest_path = root / "data" / "processed" / "splits" / "prepare_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"]["dataset"], "electricity")
            self.assertEqual(payload["dataset"]["loader"], "npz")


if __name__ == "__main__":
    unittest.main()
