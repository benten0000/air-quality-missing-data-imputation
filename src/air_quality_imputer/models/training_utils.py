import os
from math import ceil

import torch
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


def maybe_compile_model(model, config, device: torch.device):
    if device.type != "cuda" or not bool(getattr(config, "use_torch_compile", False)):
        return model
    try:
        return torch.compile(
            model,
            mode=str(getattr(config, "compile_mode", "reduce-overhead")),
            dynamic=bool(getattr(config, "compile_dynamic", False)),
        )
    except Exception as exc:
        print(f"torch.compile unavailable, fallback to eager mode: {exc}")
        return model


def build_tensor_dataloader(
    *,
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
    num_workers = max(0, int(num_workers))
    pin_memory = bool(pin_memory) and amp_enabled
    persistent_workers = bool(persistent_workers) and num_workers > 0
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(TensorDataset(x_tensor), **kwargs)


def sample_block_feature_train_mask(
    *,
    observed_mask: torch.Tensor,
    config,
    never_mask_indices: list[int] | None = None,
) -> torch.Tensor:
    mode = str(getattr(config, "train_mask_mode", "block_feature")).lower()
    missing_rate = float(getattr(config, "train_missing_rate", 0.2))
    missing_rate = min(max(missing_rate, 0.0), 1.0)

    observed = observed_mask.bool()
    bsz, seq_len, n_feat = observed.shape
    allowed = torch.ones(n_feat, dtype=torch.bool, device=observed.device)
    if never_mask_indices:
        idx = torch.as_tensor(never_mask_indices, device=observed.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < n_feat)]
        if idx.numel() > 0:
            allowed[idx] = False
    if not allowed.any():
        return torch.zeros_like(observed, dtype=torch.bool)
    maskable_observed = observed & allowed.view(1, 1, n_feat)

    if mode == "random":
        return (torch.rand_like(observed_mask) < missing_rate) & maskable_observed
    if mode == "block":
        mode = "block_feature"
    if mode != "block_feature":
        raise ValueError(f"Unsupported train_mask_mode: {mode}")

    min_len = int(getattr(config, "train_block_min_len", 2))
    max_len = int(getattr(config, "train_block_max_len", 14))
    p_block = float(getattr(config, "train_block_missing_prob", missing_rate))
    p_feat = float(getattr(config, "train_feature_block_prob", 0.6))
    p_block = min(max(p_block, 0.0), 1.0)
    p_feat = min(max(p_feat, 0.0), 1.0)
    no_overlap = bool(getattr(config, "train_block_no_overlap", True))

    if no_overlap:
        n_blocks = max(1, (seq_len + min_len - 1) // max(1, min_len))
        lengths = torch.randint(min_len, max_len + 1, (bsz, n_blocks), device=observed.device)
        ends = torch.cumsum(lengths, dim=1)
        starts = torch.cat(
            [torch.zeros((bsz, 1), device=observed.device, dtype=ends.dtype), ends[:, :-1]],
            dim=1,
        )
        starts = starts.clamp(max=seq_len).long()
        ends = ends.clamp(max=seq_len).long()
    else:
        avg = max(1.0, (min_len + max_len) / 2.0)
        n_blocks = max(1, int(round(seq_len / avg)))
        lengths = torch.randint(min_len, max_len + 1, (bsz, n_blocks), device=observed.device).clamp(max=seq_len)
        start_max = (seq_len - lengths + 1).clamp(min=1).float()
        starts = torch.floor(torch.rand((bsz, n_blocks), device=observed.device) * start_max).long()
        ends = starts + lengths

    block_active = (torch.rand((bsz, n_blocks), device=observed.device) < p_block) & (starts < seq_len)
    feat_sel = (torch.rand((bsz, n_blocks, n_feat), device=observed.device) < p_feat) & allowed.view(1, 1, n_feat)
    active_feat = feat_sel & block_active.unsqueeze(-1)

    t = torch.arange(seq_len, device=observed.device).view(1, 1, seq_len, 1)
    in_block = (t >= starts.view(bsz, n_blocks, 1, 1)) & (t < ends.view(bsz, n_blocks, 1, 1))
    mit = (in_block & active_feat.view(bsz, n_blocks, 1, n_feat)).any(dim=1) & maskable_observed

    target = int(round(missing_rate * maskable_observed.sum().item()))
    current = int(mit.sum().item())
    if current == target:
        return mit
    flat = mit.view(-1)
    if current > target:
        if target <= 0:
            flat.zero_()
            return mit
        active = torch.nonzero(flat, as_tuple=False).squeeze(1)
        drop = active.numel() - target
        if drop > 0:
            flat[active[torch.randperm(active.numel(), device=mit.device)[:drop]]] = False
        return mit

    avail = torch.nonzero((maskable_observed & ~mit).view(-1), as_tuple=False).squeeze(1)
    need = min(target - current, int(avail.numel()))
    if need > 0:
        flat[avail[torch.randperm(avail.numel(), device=mit.device)[:need]]] = True
    return mit
