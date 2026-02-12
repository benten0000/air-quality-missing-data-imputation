from __future__ import annotations

import warnings

from air_quality_imputer.pipeline.evaluate_models import main as evaluate_models_main


def main() -> None:
    warnings.warn(
        "air_quality_imputer.training.evaluate_imputer is deprecated. "
        "Use `aqi-evaluate-models --params configs/pipeline/params.yaml` or `dvc repro`.",
        DeprecationWarning,
        stacklevel=2,
    )
    evaluate_models_main()


if __name__ == "__main__":
    main()
