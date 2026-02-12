from __future__ import annotations

import warnings

from air_quality_imputer.pipeline.train_models import main as train_models_main


def main() -> None:
    warnings.warn(
        "air_quality_imputer.training.train_imputer is deprecated. "
        "Use `aqi-train-models --params configs/pipeline/params.yaml` or `dvc repro`.",
        DeprecationWarning,
        stacklevel=2,
    )
    train_models_main()


if __name__ == "__main__":
    main()
