from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from air_quality_imputer.models.transformer_imputer import TransformerConfig, TransformerImputer


class TransformerImputerTests(unittest.TestCase):
    def test_impute_does_not_force_cuda_when_available(self):
        cfg = TransformerConfig(block_size=4, n_features=3, d_model=16, n_layer=1, n_head=2)
        model = TransformerImputer(cfg).to(torch.device("cpu"))

        x = np.zeros((2, 4, 3), dtype=np.float32)
        x[0, 0, 0] = np.nan

        with patch("torch.cuda.is_available", return_value=True):
            out = model.impute({"X": x, "never_mask_feature_indices": None})
        self.assertIsNotNone(out)
        self.assertEqual(model.feature_proj.weight.device.type, "cpu")

    def test_missing_value_is_used_before_projection(self):
        cfg = TransformerConfig(block_size=4, n_features=4, d_model=4, n_layer=0, n_head=1)
        model = TransformerImputer(cfg)

        model.transformer = nn.ModuleList([])
        model.ln_f = nn.Identity()
        model.output_head = nn.Identity()

        with torch.no_grad():
            model.feature_proj.weight.zero_()
            model.feature_proj.bias.zero_()
            model.feature_proj.weight.copy_(torch.eye(4))

            model.mask_proj.weight.zero_()
            model.mask_proj.bias.zero_()

            model.gate_proj.weight.zero_()
            model.gate_proj.bias.fill_(50.0)  # sigmoid ~ 1 -> content branch

            model.missing_value.copy_(torch.tensor([5.0, 7.0, 11.0, 13.0]))

        x = torch.tensor([[[2.0, 0.0, 4.0, 6.0]]], dtype=torch.float32)
        mask = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]], dtype=torch.float32)
        out = model(x, mask).detach().cpu().numpy()

        self.assertAlmostEqual(float(out[0, 0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(out[0, 0, 1]), 7.0, places=5)
        self.assertAlmostEqual(float(out[0, 0, 2]), 4.0, places=5)
        self.assertAlmostEqual(float(out[0, 0, 3]), 6.0, places=5)


if __name__ == "__main__":
    unittest.main()
