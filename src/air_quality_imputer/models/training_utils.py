import os
from math import ceil
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def maybe_compile_model(model: nn.Module, config, device: torch.device) -> nn.Module:
    use_torch_compile = bool(config.use_torch_compile)
    if device.type != "cuda" or not use_torch_compile:
        return model
    try:
        compile_mode = str(config.compile_mode)
        compile_dynamic = bool(config.compile_dynamic)
        compiled = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)
        return cast(nn.Module, compiled)
    except Exception as exc:
        print(f"torch.compile unavailable, fallback to eager mode: {exc}")
        return model


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _build_allowed_feature_mask(
    n_feat: int,
    device: torch.device,
    never_mask_indices: list[int] | None,
) -> torch.Tensor:
    allowed = torch.ones(n_feat, dtype=torch.bool, device=device)
    if not never_mask_indices:
        return allowed
    idx = torch.as_tensor(never_mask_indices, device=device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < n_feat)]
    if idx.numel() > 0:
        allowed[idx] = False
    return allowed


def _adjust_mask_to_target(
    mit_mask: torch.Tensor,
    maskable_observed: torch.Tensor,
    missing_rate: float,
) -> torch.Tensor:
    target_missing = int(round(missing_rate * maskable_observed.sum().item()))
    current_missing = int(mit_mask.sum().item())
    if current_missing == target_missing:
        return mit_mask

    flat_mask = mit_mask.view(-1)
    if current_missing > target_missing:
        active = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        if target_missing <= 0:
            flat_mask.zero_()
            return mit_mask
        if target_missing < active.numel():
            drop = active.numel() - target_missing
            drop_idx = active[torch.randperm(active.numel(), device=mit_mask.device)[:drop]]
            flat_mask[drop_idx] = False
        return mit_mask

    available = torch.nonzero((maskable_observed & ~mit_mask).view(-1), as_tuple=False).squeeze(1)
    need = target_missing - current_missing
    if need > 0 and available.numel() > 0:
        add_idx = available[torch.randperm(available.numel(), device=mit_mask.device)[: min(need, available.numel())]]
        flat_mask[add_idx] = True
    return mit_mask


def build_tensor_dataloader(
    x_tensor: torch.Tensor,
    batch_size: int,
    amp_enabled: bool,
    num_workers: int = -1,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    if num_workers < 0:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(8, ceil(cpu_count / 2)))
    num_workers = max(0, num_workers)

    pin_memory = bool(pin_memory) and amp_enabled
    persistent_workers = bool(persistent_workers) and num_workers > 0
    prefetch_factor = max(2, int(prefetch_factor))

    dataset = TensorDataset(x_tensor)
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


def sample_block_feature_train_mask(
    observed_mask: torch.Tensor,
    config,
    never_mask_indices: list[int] | None = None,
) -> torch.Tensor:
    mode = str(config.train_mask_mode)
    missing_rate = _clamp01(float(config.train_missing_rate))

    observed = observed_mask.bool()
    bsz, seq_len, n_feat = observed.shape
    allowed_feature_mask = _build_allowed_feature_mask(
        n_feat=n_feat,
        device=observed.device,
        never_mask_indices=never_mask_indices,
    )
    if not allowed_feature_mask.any():
        return torch.zeros_like(observed, dtype=torch.bool)
    maskable_observed = observed & allowed_feature_mask.view(1, 1, n_feat)

    if mode == "random":
        return (torch.rand_like(observed_mask) < missing_rate) & maskable_observed
    if mode != "block_feature":
        raise ValueError(f"Unsupported train_mask_mode: {mode}")

    min_len = int(config.train_block_min_len)
    max_len = int(config.train_block_max_len)
    if min_len <= 0 or max_len <= 0:
        raise ValueError("train_block_min_len and train_block_max_len must be positive")
    if min_len > max_len:
        raise ValueError("train_block_min_len must be <= train_block_max_len")

    p_block_missing = _clamp01(float(config.train_block_missing_prob))
    p_feature_missing = _clamp01(float(config.train_feature_block_prob))
    no_overlap = bool(config.train_block_no_overlap)

    if no_overlap:
        n_blocks = max(1, (seq_len + min_len - 1) // min_len)
        block_lengths = torch.randint(min_len, max_len + 1, (bsz, n_blocks), device=observed.device)
        block_ends = torch.cumsum(block_lengths, dim=1)
        block_starts = torch.cat(
            [
                torch.zeros((bsz, 1), device=observed.device, dtype=block_ends.dtype),
                block_ends[:, :-1],
            ],
            dim=1,
        )
        block_starts = block_starts.clamp(max=seq_len).long()
        block_ends = block_ends.clamp(max=seq_len).long()
    else:
        avg_block_len = max(1.0, (min_len + max_len) / 2.0)
        n_blocks = max(1, int(round(seq_len / avg_block_len)))
        block_lengths = torch.randint(min_len, max_len + 1, (bsz, n_blocks), device=observed.device).clamp(max=seq_len)
        start_max = (seq_len - block_lengths + 1).clamp(min=1).float()
        block_starts = torch.floor(torch.rand((bsz, n_blocks), device=observed.device) * start_max).long()
        block_ends = block_starts + block_lengths

    block_valid = block_starts < seq_len
    block_active = (torch.rand((bsz, n_blocks), device=observed.device) < p_block_missing) & block_valid
    feature_sel = (
        torch.rand((bsz, n_blocks, n_feat), device=observed.device) < p_feature_missing
    ) & allowed_feature_mask.view(1, 1, n_feat)
    active_feat = feature_sel & block_active.unsqueeze(-1)

    time_idx = torch.arange(seq_len, device=observed.device).view(1, 1, seq_len, 1)
    time_sel = (time_idx >= block_starts.view(bsz, n_blocks, 1, 1)) & (time_idx < block_ends.view(bsz, n_blocks, 1, 1))
    mit_mask = (time_sel & active_feat.view(bsz, n_blocks, 1, n_feat)).any(dim=1)
    mit_mask = mit_mask & maskable_observed

    return _adjust_mask_to_target(
        mit_mask=mit_mask,
        maskable_observed=maskable_observed,
        missing_rate=missing_rate,
    )
