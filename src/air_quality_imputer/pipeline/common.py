from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("configs/pipeline/params.yaml"),
        help="Path to pipeline params.yaml used by DVC.",
    )
    return parser


def load_params(params_path: Path) -> DictConfig:
    cfg = OmegaConf.load(params_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {params_path}, got {type(cfg)!r}")
    return cfg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, station: str, model_name: str) -> int:
    key = f"{base_seed}:{station}:{model_name}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return {str(key): item for key, item in cfg.items()}
    return {}


def get_model_cfg_from_params(cfg: DictConfig, model_name: str) -> DictConfig:
    models_cfg = cfg.get("models")
    if models_cfg is None or model_name not in models_cfg:
        raise KeyError(f"Missing models.{model_name} in pipeline params.")
    model_cfg = models_cfg[model_name]
    if isinstance(model_cfg, DictConfig):
        return model_cfg
    created = OmegaConf.create(model_cfg)
    if not isinstance(created, DictConfig):
        raise TypeError(f"models.{model_name} must be a mapping config.")
    return created
