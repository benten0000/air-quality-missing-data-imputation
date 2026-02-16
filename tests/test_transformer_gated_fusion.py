from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from air_quality_imputer.models.transformer_imputer import TransformerConfig, TransformerImputer


class TransformerGatedFusionTests(unittest.TestCase):
    def _make_identity_model(self, *, gate_bias: float) -> TransformerImputer:
        cfg = TransformerConfig(
            block_size=4,
            n_features=3,
            d_model=3,
            n_layer=0,
            n_head=1,
        )
        model = TransformerImputer(cfg)
        model.transformer = nn.ModuleList([])
        model.ln_f = nn.Identity()
        model.output_head = nn.Identity()

        with torch.no_grad():
            model.feature_proj.weight.zero_()
            model.feature_proj.bias.zero_()
            model.feature_proj.weight.copy_(torch.eye(3))

            model.mask_proj.weight.zero_()
            model.mask_proj.bias.zero_()

            model.gate_proj.weight.zero_()
            model.gate_proj.bias.fill_(gate_bias)

        return model

    def test_missing_value_is_used_when_gate_prefers_x(self):
        model = self._make_identity_model(gate_bias=50.0)  # sigmoid ~ 1
        with torch.no_grad():
            model.missing_value.copy_(torch.tensor([10.0, 20.0, 30.0]))

        x = np.zeros((1, 4, 3), dtype=np.float32)
        mask = np.ones_like(x, dtype=np.float32)
        mask[:, 0, 1] = 0.0  # feature 1 missing at t=0

        out = model(torch.tensor(x), torch.tensor(mask))
        out_np = out.detach().cpu().numpy()
        self.assertAlmostEqual(float(out_np[0, 0, 1]), 20.0, places=5)

    def test_gate_can_ignore_x_and_use_mask_branch(self):
        model = self._make_identity_model(gate_bias=-50.0)  # sigmoid ~ 0
        with torch.no_grad():
            model.missing_value.copy_(torch.tensor([10.0, 20.0, 30.0]))

        x = np.zeros((1, 4, 3), dtype=np.float32)
        mask = np.ones_like(x, dtype=np.float32)
        mask[:, 0, 1] = 0.0

        out = model(torch.tensor(x), torch.tensor(mask))
        out_np = out.detach().cpu().numpy()
        self.assertAlmostEqual(float(out_np[0, 0, 1]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
