import hydra
from omegaconf import DictConfig

from air_quality_imputer.training.evaluate_imputer import run


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
