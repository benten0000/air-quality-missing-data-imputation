from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from air_quality_imputer.data.electricity_dataset import prepare_electricity_csv, prepare_electricity_npz


class ElectricityDatasetTests(unittest.TestCase):
    def test_prepare_electricity_csv_from_local_zip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            zip_path = cache_dir / "LD2011_2014.txt.zip"

            sample_content = "\n".join(
                [
                    '"";"MT_001";"MT_002";"MT_003"',
                    '"2011-01-01 00:15:00";1,0;2,0;3,0',
                    '"2011-01-01 00:30:00";3,0;4,0;5,0',
                    '"2011-01-01 00:45:00";5,0;6,0;7,0',
                    '"2011-01-01 01:00:00";7,0;8,0;9,0',
                ]
            )
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("LD2011_2014.txt", sample_content)

            output_csv = root / "data" / "datasets" / "electricity" / "raw" / "electricity.csv"
            csv_path, metadata_path = prepare_electricity_csv(
                output_csv=output_csv,
                cache_dir=cache_dir,
                url="unused",
                force_download=False,
                force_extract=True,
                clients=["MT_001", "MT_002"],
                start_client=1,
                n_clients=2,
                resample_frequency="1h",
                max_rows=None,
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(metadata_path.exists())

            df = pd.read_csv(csv_path)
            self.assertListEqual(list(df.columns), ["datetime", "MT_001", "MT_002"])
            self.assertEqual(len(df), 2)
            self.assertAlmostEqual(float(df.iloc[0]["MT_001"]), 3.0, places=6)
            self.assertAlmostEqual(float(df.iloc[0]["MT_002"]), 4.0, places=6)

    def test_prepare_electricity_npz_from_local_zip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            zip_path = cache_dir / "LD2011_2014.txt.zip"

            sample_content = "\n".join(
                [
                    '"";"MT_001";"MT_002"',
                    '"2011-01-01 00:15:00";1,0;2,0',
                    '"2011-01-01 00:30:00";3,0;4,0',
                    '"2011-01-01 00:45:00";5,0;6,0',
                    '"2011-01-01 01:00:00";7,0;8,0',
                ]
            )
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("LD2011_2014.txt", sample_content)

            output_npz = root / "data" / "datasets" / "electricity" / "npz" / "electricity.npz"
            npz_path, metadata_path = prepare_electricity_npz(
                output_npz=output_npz,
                cache_dir=cache_dir,
                url="unused",
                force_download=False,
                force_extract=True,
                clients=["MT_001", "MT_002"],
                start_client=1,
                n_clients=2,
                resample_frequency="1h",
                max_rows=None,
            )

            self.assertTrue(npz_path.exists())
            self.assertTrue(metadata_path.exists())

            payload = np.load(npz_path)
            self.assertIn("X", payload.files)
            self.assertIn("datetime", payload.files)
            self.assertIn("feature_names", payload.files)
            self.assertEqual(payload["X"].shape, (2, 2))
            self.assertListEqual(payload["feature_names"].tolist(), ["MT_001", "MT_002"])


if __name__ == "__main__":
    unittest.main()
