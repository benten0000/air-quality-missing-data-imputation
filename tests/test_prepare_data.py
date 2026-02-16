from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from air_quality_imputer import exceptions
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
                        "never_mask_features": [],
                        "block_size": 4,
                        "step_size": 1,
                        "seed": 42,
                        "train_split_ratio": 0.7,
                        "val_split_ratio": 0.1,
                        "test_split_ratio": 0.2,
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
                    },
                }
            )

            run(cfg)

            split_root = root / "data" / "processed" / "splits" / "all_stations"
            self.assertTrue((split_root / "windows.npz").exists())
            self.assertTrue((root / "data" / "processed" / "splits" / "prepare_manifest.json").exists())
            w = np.load(split_root / "windows.npz")
            # Windows are stored as [n_windows, block_size, n_features] throughout the pipeline.
            self.assertEqual(w["X_train"].ndim, 3)
            self.assertEqual(int(w["X_train"].shape[1]), 4)
            self.assertEqual(int(w["X_train"].shape[2]), 9)

    def test_prepare_stage_uses_all_csv_columns_when_features_empty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            station = "electricity"
            rows = 72
            pd.DataFrame(
                {
                    "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
                    "station": ["A"] * rows,
                    "MT_001": np.linspace(1, 3, rows),
                    "MT_002": np.linspace(10, 20, rows),
                    "MT_003": np.linspace(100, 200, rows),
                }
            ).to_csv(raw_dir / f"{station}.csv", index=False)

            cfg = OmegaConf.create(
                {
                    "experiment": {
                        "dataset": "electricity",
                        "stations": [station],
                        "features": [],
                        "never_mask_features": [],
                        "block_size": 12,
                        "step_size": 1,
                        "seed": 42,
                        "train_split_ratio": 0.7,
                        "val_split_ratio": 0.1,
                        "test_split_ratio": 0.2,
                    },
                    "dataset": {
                        "definitions": {
                            "electricity": {
                                "loader": "air_quality_csv",
                                "data_dir": str(raw_dir),
                                # Should be ignored in favor of all columns inferred from CSV.
                                "feature_names": ["MT_001"],
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
                    },
                }
            )

            run(cfg)

            split_root = root / "data" / "processed" / "splits" / station
            self.assertTrue((split_root / "windows.npz").exists())
            manifest_path = root / "data" / "processed" / "splits" / "prepare_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"]["dataset"], "electricity")
            self.assertEqual(payload["dataset"]["loader"], "air_quality_csv")
            self.assertEqual(payload["resolved_features"], ["MT_001", "MT_002", "MT_003"])

    def test_prepare_stage_rejects_non_csv_loader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
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
                        "train_split_ratio": 0.7,
                        "val_split_ratio": 0.1,
                        "test_split_ratio": 0.2,
                    },
                    "dataset": {
                        "definitions": {
                            "electricity": {
                                "loader": "npz",
                                "data_dir": str(root / "data" / "raw"),
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
                        "data_dir": str(root / "data" / "raw"),
                        "processed_dir": str(root / "data" / "processed" / "splits"),
                    },
                }
            )

            with self.assertRaises(exceptions.ValidationError):
                run(cfg)

if __name__ == "__main__":
    unittest.main()
