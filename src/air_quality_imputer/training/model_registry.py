from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf

from air_quality_imputer.models.diffusion_imputer import (
    DiffusionTransformerConfig,
    DiffusionTransformerImputer,
)
from air_quality_imputer.models.transformer_imputer import TransformerConfig, TransformerImputer


MODEL_REGISTRY = {
    "classic_transformer": {
        "config_cls": TransformerConfig,
        "model_cls": TransformerImputer,
    },
    "diffusion_transformer": {
        "config_cls": DiffusionTransformerConfig,
        "model_cls": DiffusionTransformerImputer,
    },
}


def _to_plain_dict(params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, DictConfig):
        container = OmegaConf.to_container(params, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Expected mapping-like params, got: {type(container)!r}")
        return cast(dict[str, Any], container)
    if isinstance(params, dict):
        if not all(isinstance(k, str) for k in params):
            raise TypeError("Model params keys must be strings")
        return cast(dict[str, Any], dict(params))
    if isinstance(params, Mapping):
        out: dict[str, Any] = {}
        for k, v in params.items():
            if not isinstance(k, str):
                raise TypeError("Model params keys must be strings")
            out[k] = v
        return out
    raise TypeError(f"Expected mapping-like params, got: {type(params)!r}")


def build_model_from_cfg(model_cfg: DictConfig, n_features: int, block_size: int):
    model_type = str(model_cfg.type)
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    spec = MODEL_REGISTRY[model_type]
    config_cls = spec["config_cls"]
    model_cls = spec["model_cls"]

    params = _to_plain_dict(model_cfg.params)
    params["n_features"] = n_features
    params["block_size"] = block_size

    model_config = config_cls(**params)
    model = model_cls(model_config)
    checkpoint_name = str(model_cfg.checkpoint_name)
    return model, model_config, model_type, checkpoint_name


def build_model_from_checkpoint(model_type: str, model_config):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    config_cls = MODEL_REGISTRY[model_type]["config_cls"]
    if isinstance(model_config, dict):
        model_config = config_cls(**model_config)
    model_cls = MODEL_REGISTRY[model_type]["model_cls"]
    return model_cls(model_config)


def config_to_dict(model_config) -> dict[str, Any]:
    if is_dataclass(model_config) and not isinstance(model_config, type):
        return asdict(model_config)
    if isinstance(model_config, dict):
        if not all(isinstance(k, str) for k in model_config):
            raise TypeError("Model config keys must be strings")
        return cast(dict[str, Any], dict(model_config))
    if isinstance(model_config, Mapping):
        out: dict[str, Any] = {}
        for k, v in model_config.items():
            if not isinstance(k, str):
                raise TypeError("Model config keys must be strings")
            out[k] = v
        return out
    raise TypeError(f"Unsupported model_config type: {type(model_config)!r}")


def load_model_cfg_by_name(model_name: str) -> DictConfig:
    model_cfg_path = Path(__file__).resolve().parent / "conf" / "model" / f"{model_name}.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Missing model config: {model_cfg_path}")
    cfg = OmegaConf.load(model_cfg_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {model_cfg_path}, got {type(cfg)!r}")
    return cfg
