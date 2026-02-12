from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _extract_imputation(output: Any) -> np.ndarray | None:
    arr_raw: Any | None = None
    if isinstance(output, dict):
        if "imputation" in output:
            arr_raw = output["imputation"]
        elif "X_imputed" in output:
            arr_raw = output["X_imputed"]
    elif isinstance(output, np.ndarray):
        arr_raw = output

    if arr_raw is None:
        return None
    arr_np = np.asarray(arr_raw)
    if arr_np.ndim == 4:
        if arr_np.shape[1] == 1:
            arr_np = arr_np[:, 0]
        elif arr_np.shape[0] == 1:
            arr_np = arr_np[0]
        else:
            arr_np = np.median(arr_np, axis=1)
    return arr_np.astype(np.float32, copy=False)


class _PyPOTSBase:
    def __init__(self):
        self.model = None

    def _device_name(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def state_dict(self) -> dict[str, Any]:
        return {"serialized_model": self.model}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model = state_dict["serialized_model"]

    def to(self, device: torch.device):
        del device
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def number_of_params(self) -> int:
        if self.model is None:
            return 0
        torch_model = getattr(self.model, "model", None)
        if torch_model is None:
            return 0
        return int(sum(p.numel() for p in torch_model.parameters()))

    def impute(self, dataset: dict[str, np.ndarray]) -> np.ndarray | None:
        if self.model is None:
            return None
        output = self.model.impute({"X": dataset["X"]})
        return _extract_imputation(output)


@dataclass
class SAITSConfig:
    block_size: int = 24
    n_features: int = 10
    n_layers: int = 3
    d_model: int = 128
    n_head: int = 4
    dropout: float = 0.1
    d_ffn: int | None = None
    ORT_weight: int = 1
    MIT_weight: int = 1


class SAITSImputer(_PyPOTSBase):
    def __init__(self, config: SAITSConfig):
        super().__init__()
        self.config = config
        self._build_model(epochs=300, batch_size=128, patience=250, lr=1e-3)

    def _build_model(self, epochs: int, batch_size: int, patience: int, lr: float) -> None:
        from pypots.imputation import SAITS
        from pypots.optim.adam import Adam

        d_k = self.config.d_model // self.config.n_head
        d_v = self.config.d_model // self.config.n_head
        self.model = SAITS(
            n_steps=self.config.block_size,
            n_features=self.config.n_features,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            n_heads=self.config.n_head,
            d_k=d_k,
            d_v=d_v,
            d_ffn=self.config.d_ffn or (4 * self.config.d_model),
            dropout=self.config.dropout,
            ORT_weight=self.config.ORT_weight,
            MIT_weight=self.config.MIT_weight,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            optimizer=Adam(lr=lr),
            device=self._device_name(),
        )

    def fit(
        self,
        dataset: dict[str, np.ndarray],
        epochs: int = 300,
        batch_size: int = 128,
        initial_lr: float = 1e-3,
        patience: int = 250,
        min_delta: float = 0.0,
        validation_data: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float | None]:
        if min_delta != 0.0:
            print("[INFO] SAITS backend does not expose min_delta; using patience-only early stopping.")
        self._build_model(epochs=epochs, batch_size=batch_size, patience=patience, lr=initial_lr)
        assert self.model is not None
        train_set = {"X": dataset["X"]}
        val_set = None
        if validation_data is not None and "X" in validation_data and "X_ori" in validation_data:
            val_set = {"X": validation_data["X"], "X_ori": validation_data["X_ori"]}
        if val_set is not None:
            self.model.fit(train_set, val_set)
        else:
            self.model.fit(train_set)
        return {"best_loss": None, "stopped_epoch": None}
