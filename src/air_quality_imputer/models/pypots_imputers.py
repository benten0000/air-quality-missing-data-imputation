from dataclasses import dataclass
import inspect
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
    attn_dropout: float = 0.0
    diagonal_attention_mask: bool = True
    d_ffn: int | None = None
    ORT_weight: float = 1.0
    MIT_weight: float = 1.0
    learning_rate: float | None = None
    training_loss: str = "MAE"
    validation_metric: str = "MSE"
    num_workers: int = 0
    saving_path: str | None = None
    model_saving_strategy: str | None = "best"
    verbose: bool = True
    optimizer: str = "adam"
    optimizer_weight_decay: float = 0.0
    optimizer_betas: list[float] | None = None
    optimizer_eps: float | None = None


class SAITSImputer(_PyPOTSBase):
    def __init__(self, config: SAITSConfig):
        super().__init__()
        self.config = config
        self._build_model(epochs=300, batch_size=128, patience=250, lr=1e-3)

    @staticmethod
    def _filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        params = inspect.signature(callable_obj).parameters
        return {key: value for key, value in kwargs.items() if key in params and value is not None}

    def _build_optimizer(self, lr: float):
        effective_lr = float(self.config.learning_rate) if self.config.learning_rate is not None else float(lr)
        optimizer_name = str(self.config.optimizer).lower()
        common_kwargs: dict[str, Any] = {
            "lr": effective_lr,
            "weight_decay": float(self.config.optimizer_weight_decay),
            "eps": self.config.optimizer_eps,
        }
        if self.config.optimizer_betas is not None and len(self.config.optimizer_betas) == 2:
            common_kwargs["betas"] = tuple(float(x) for x in self.config.optimizer_betas)

        if optimizer_name == "adam":
            from pypots.optim.adam import Adam

            return Adam(**self._filtered_kwargs(Adam, common_kwargs))
        if optimizer_name == "adamw":
            try:
                from pypots.optim.adamw import AdamW
            except Exception as exc:
                raise ValueError("optimizer=adamw is not available in current PyPOTS installation.") from exc
            return AdamW(**self._filtered_kwargs(AdamW, common_kwargs))
        raise ValueError(f"Unsupported SAITS optimizer: {self.config.optimizer}")

    @staticmethod
    def _resolve_criterion(name: str):
        from pypots.nn.modules import loss as loss_mod

        allowed = {
            "MAE": loss_mod.MAE,
            "MSE": loss_mod.MSE,
            "RMSE": loss_mod.RMSE,
            "MRE": loss_mod.MRE,
            "NLL": loss_mod.NLL,
            "CROSSENTROPY": loss_mod.CrossEntropy,
        }
        key = str(name).replace("-", "").replace("_", "").upper()
        if key not in allowed:
            raise ValueError(f"Unsupported SAITS criterion: {name}. Allowed: {sorted(allowed.keys())}")
        return allowed[key]

    def _build_model(self, epochs: int, batch_size: int, patience: int, lr: float) -> None:
        from pypots.imputation import SAITS

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
            attn_dropout=self.config.attn_dropout,
            diagonal_attention_mask=self.config.diagonal_attention_mask,
            ORT_weight=self.config.ORT_weight,
            MIT_weight=self.config.MIT_weight,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            training_loss=self._resolve_criterion(self.config.training_loss),
            validation_metric=self._resolve_criterion(self.config.validation_metric),
            optimizer=self._build_optimizer(lr=lr),
            num_workers=int(self.config.num_workers),
            device=self._device_name(),
            saving_path=self.config.saving_path,
            model_saving_strategy=self.config.model_saving_strategy,
            verbose=bool(self.config.verbose),
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
