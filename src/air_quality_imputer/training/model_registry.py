from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf

from air_quality_imputer.models.pypots_imputers import SAITSConfig, SAITSImputer
from air_quality_imputer.models.transformer_imputer import TransformerConfig, TransformerImputer


MODEL_REGISTRY = {
    "classic_transformer": {
        "config_cls": TransformerConfig,
        "model_cls": TransformerImputer,
    },
    "saits": {
        "config_cls": SAITSConfig,
        "model_cls": SAITSImputer,
    },
}


def _mapping_to_str_key_dict(value: Mapping[Any, Any]) -> dict[str, Any]:
    if not all(isinstance(k, str) for k in value):
        raise TypeError("Model config keys must be strings")
    return {cast(str, k): v for k, v in value.items()}


def _to_plain_dict(params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, DictConfig):
        container = OmegaConf.to_container(params, resolve=True)
        if not isinstance(container, Mapping):
            raise TypeError(f"Expected mapping-like params, got: {type(container)!r}")
        return _mapping_to_str_key_dict(container)
    if isinstance(params, Mapping):
        return _mapping_to_str_key_dict(params)
    raise TypeError(f"Expected mapping-like params, got: {type(params)!r}")


def build_model_from_cfg(
    model_cfg: DictConfig,
    n_features: int,
    block_size: int,
    runtime_params: Mapping[str, Any] | None = None,
):
    model_type = str(model_cfg.type)
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    spec = MODEL_REGISTRY[model_type]
    config_cls = spec["config_cls"]
    model_cls = spec["model_cls"]

    params = _to_plain_dict(model_cfg.params)
    params["n_features"] = n_features
    params["block_size"] = block_size
    if runtime_params:
        dataclass_fields = getattr(config_cls, "__dataclass_fields__", {})
        allowed_keys = set(dataclass_fields.keys()) if isinstance(dataclass_fields, dict) else set()
        for key, value in runtime_params.items():
            if key in allowed_keys:
                params[key] = value

    model_config = config_cls(**params)
    model = model_cls(model_config)
    checkpoint_name = str(model_cfg.checkpoint_name)
    return model, model_config, model_type, checkpoint_name


def build_model_from_checkpoint(model_type: str, model_config):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    config_cls = MODEL_REGISTRY[model_type]["config_cls"]
    if isinstance(model_config, Mapping):
        model_config = config_cls(**model_config)
    model_cls = MODEL_REGISTRY[model_type]["model_cls"]
    return model_cls(model_config)


def config_to_dict(model_config) -> dict[str, Any]:
    if is_dataclass(model_config) and not isinstance(model_config, type):
        return asdict(model_config)
    if isinstance(model_config, Mapping):
        return _mapping_to_str_key_dict(model_config)
    raise TypeError(f"Unsupported model_config type: {type(model_config)!r}")


def load_model_cfg_by_name(model_name: str) -> DictConfig:
    legacy_path = Path(__file__).resolve().parent / "conf" / "model" / f"{model_name}.yaml"
    repo_root = Path(__file__).resolve().parents[3]
    centralized_path = repo_root / "configs" / "legacy" / "hydra" / "model" / f"{model_name}.yaml"
    model_cfg_path = centralized_path if centralized_path.exists() else legacy_path
    if not model_cfg_path.exists():
        raise FileNotFoundError(
            f"Missing model config for {model_name!r}. "
            f"Tried: {centralized_path} and {legacy_path}"
        )
    cfg = OmegaConf.load(model_cfg_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {model_cfg_path}, got {type(cfg)!r}")
    return cfg
