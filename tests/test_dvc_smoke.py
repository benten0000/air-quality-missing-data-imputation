from __future__ import annotations

import shutil
import subprocess
import unittest


class DVCSmokeTests(unittest.TestCase):
    @unittest.skipIf(shutil.which("dvc") is None, "dvc CLI not installed in environment")
    def test_dvc_pipeline_dry_repro(self):
        result = subprocess.run(["dvc", "repro", "--dry"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
