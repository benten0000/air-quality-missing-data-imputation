from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from air_quality_imputer.models.training_utils import (
    build_tensor_dataloader,
    configure_cuda_runtime,
    maybe_compile_model,
    sample_block_feature_train_mask,
)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 5000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        positions = torch.arange(max_len).unsqueeze(1)
        angle = positions * theta.unsqueeze(0)
        self.register_buffer("cos_cached", torch.cos(angle))
        self.register_buffer("sin_cached", torch.sin(angle))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, seq_len, _ = x.size()
        cos_cached = cast(torch.Tensor, self.cos_cached)
        sin_cached = cast(torch.Tensor, self.sin_cached)
        cos = cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return x_rot


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def get_diagonal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    diag = torch.eye(seq_len, device=device, dtype=dtype)
    return diag * -1e9


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.head_dim = config.d_model // config.n_head
        self.rotary = RotaryPositionalEncoding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, channels = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        attn_mask = get_diagonal_mask(seq_len, x.device, q.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.d_model
        self.c_fc = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.c_gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x) * F.silu(self.c_gate(x))
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ff = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


@dataclass
class TransformerConfig:
    block_size: int = 24
    n_features: int = 10

    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    dropout: float = 0.1
    bias: bool = True
    norm_eps: float = 1e-6

    train_mask_mode: str = "block_feature"
    train_missing_rate: float = 0.25
    train_block_min_len: int = 2
    train_block_max_len: int = 14
    train_block_missing_prob: float = 0.35
    train_feature_block_prob: float = 0.6
    train_block_no_overlap: bool = True

    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = False
    optimizer_fused: bool = True
    optimizer_weight_decay: float = 0.01
    scheduler_warmup_ratio: float = 0.125
    scheduler_warmup_start_factor: float = 0.3
    scheduler_min_lr: float = 1e-7
    grad_clip_norm: float = 0.5
    dataloader_num_workers: int = -1
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: bool = True


class TransformerImputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feature_proj = nn.Linear(config.n_features, config.d_model)
        self.mask_proj = nn.Linear(config.n_features, config.d_model, bias=config.bias)
        self.combined_proj = nn.Linear(2 * config.d_model, config.d_model)

        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_head = nn.Linear(config.d_model, config.n_features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_embedded = self.feature_proj(x)
        mask_embedded = self.mask_proj(mask)
        combined = torch.cat([x_embedded, mask_embedded], dim=-1)
        combined = self.combined_proj(combined)

        for block in self.transformer:
            combined = block(combined)

        combined = self.ln_f(combined)
        return self.output_head(combined)

    def _sample_train_mask(self, observed_mask: torch.Tensor, never_mask_indices: list[int] | None = None) -> torch.Tensor:
        return sample_block_feature_train_mask(
            observed_mask=observed_mask,
            config=self.config,
            never_mask_indices=never_mask_indices,
        )

    def fit(
        self,
        dataset,
        epochs=300,
        batch_size=128,
        initial_lr=1e-3,
        patience=250,
        min_delta=0.0,
        validation_data=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        cfg = self.config

        compiled_model = maybe_compile_model(self, cfg, device)
        configure_cuda_runtime(device)

        x_tensor = torch.tensor(dataset["X"], dtype=torch.float32)

        amp = device.type == "cuda"
        scaler = torch.GradScaler(device.type, enabled=amp)

        use_fused = bool(cfg.optimizer_fused) and amp
        weight_decay = float(cfg.optimizer_weight_decay)
        try:
            optimizer = torch.optim.AdamW(
                compiled_model.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay,
                fused=use_fused,
            )
        except Exception:
            optimizer = torch.optim.AdamW(
                compiled_model.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay,
                fused=False,
            )

        warmup_ratio = min(max(float(cfg.scheduler_warmup_ratio), 0.0), 1.0)
        warmup_epochs = max(1, int(round(epochs * warmup_ratio)))
        warmup_start_factor = float(cfg.scheduler_warmup_start_factor)
        eta_min = float(cfg.scheduler_min_lr)
        grad_clip_norm = float(cfg.grad_clip_norm)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            total_iters=warmup_epochs,
        )
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])

        loader = build_tensor_dataloader(
            x_tensor=x_tensor,
            batch_size=batch_size,
            amp_enabled=amp,
            num_workers=int(cfg.dataloader_num_workers),
            prefetch_factor=int(cfg.dataloader_prefetch_factor),
            persistent_workers=bool(cfg.dataloader_persistent_workers),
            pin_memory=bool(cfg.dataloader_pin_memory),
            shuffle=True,
        )

        never_mask_indices = dataset.get("never_mask_feature_indices")
        best = float("inf")
        patience_counter = 0
        use_val = validation_data is not None

        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch
            compiled_model.train()
            total_loss = 0.0
            n_batches = 0

            for (batch_x_cpu,) in loader:
                batch_x = batch_x_cpu.to(device, non_blocking=amp)
                original_mask = (~torch.isnan(batch_x)).float()
                mit_mask = self._sample_train_mask(original_mask, never_mask_indices=never_mask_indices)
                if not mit_mask.any():
                    continue

                x_in = batch_x.clone()
                x_in[mit_mask] = 0
                input_mask = original_mask.clone()
                input_mask[mit_mask] = 0

                torch.nan_to_num_(x_in, 0.0)
                target = torch.nan_to_num(batch_x.clone(), 0.0)

                with torch.autocast(device_type=device.type, enabled=amp):
                    preds = compiled_model(x_in, input_mask)
                    mit_loss = torch.abs(preds[mit_mask] - target[mit_mask]).mean() if mit_mask.sum() > 0 else x_in.new_zeros(())
                    obs_mask = original_mask.bool() & ~mit_mask
                    ort_loss = torch.abs(preds[obs_mask] - target[obs_mask]).mean() if obs_mask.sum() > 0 else x_in.new_zeros(())
                    loss = mit_loss + ort_loss

                if torch.isfinite(loss).item() and loss.item() > 0:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += float(loss.item())
                    n_batches += 1

            if n_batches == 0:
                continue

            train_loss = total_loss / n_batches
            val_loss = self._validate_imputation(validation_data) if validation_data else None
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            if epoch % 10 == 0 or epoch < 10:
                if val_loss is not None and not np.isnan(val_loss):
                    print(f"Epoch {epoch:3d}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
                else:
                    print(f"Epoch {epoch:3d}, Train: {train_loss:.6f}, LR: {current_lr:.2e}")

            metric = val_loss if (use_val and val_loss is not None and not np.isnan(val_loss)) else train_loss
            if metric < best - min_delta:
                best = metric
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Ucenje se je predcasno ustavilo pri epohi {epoch} (najnizja izguba: {best:.6f})")
                break

        print(f"Ucenje zakljuceno. Koncna vrednost izgube: {best:.6f}")
        return {
            "best_loss": float(best) if np.isfinite(best) else None,
            "stopped_epoch": int(last_epoch),
        }

    def _validate_imputation(self, validation_data):
        try:
            x_val_masked = validation_data["X"]
            x_val_ori = validation_data["X_ori"]
            if x_val_masked is None or x_val_ori is None:
                return np.nan
            if len(x_val_masked) == 0 or len(x_val_ori) == 0:
                return np.nan
            missing_mask = np.isnan(x_val_masked) & ~np.isnan(x_val_ori)
            if not missing_mask.any():
                return np.nan

            dataset_masked = {"X": x_val_masked}

            self.eval()
            with torch.no_grad():
                x_imputed = self.impute(dataset_masked)
                if x_imputed is None:
                    return np.nan
                original_values = x_val_ori[missing_mask]
                imputed_values = x_imputed[missing_mask]
                if np.isnan(original_values).any() or np.isnan(imputed_values).any():
                    return np.nan
                mae = np.abs(imputed_values - original_values).mean()
                if np.isnan(mae) or np.isinf(mae):
                    return np.nan
                return mae
        except Exception as exc:
            print(f"Validacija imputacije ni uspela: {exc}")
            return np.nan
        finally:
            self.train()

    def impute(self, dataset):
        try:
            self.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)

            x = dataset["X"]
            if x is None or len(x) == 0:
                return None

            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

            amp_enabled = device.type == "cuda"
            with torch.no_grad():
                missing_mask = torch.isnan(x_tensor)
                observation_mask = (~missing_mask).float()
                x_input = torch.nan_to_num(x_tensor, 0)

                if amp_enabled:
                    with torch.autocast(device_type=device.type, enabled=True):
                        predictions = self.forward(x_input, observation_mask)
                    predictions = predictions.float()
                else:
                    predictions = self.forward(x_input, observation_mask)

                result = x_input.clone()
                result[missing_mask] = predictions[missing_mask]
                if torch.isnan(result).any():
                    print("NaN vrednosti v rezultatu zapolnjevanja")
                    return None
                return result.cpu().numpy()
        except Exception as exc:
            print(f"Napaka pri zapolnjevanju manjkajocih vrednosti: {exc}")
            return None

    def number_of_params(self):
        return sum(p.numel() for p in self.parameters())
