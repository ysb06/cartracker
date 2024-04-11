import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig


logger = logging.getLogger()

CONFIG_BASE_PATH = Path(__file__).parent / "config"
TRAINING_CONFIG = CONFIG_BASE_PATH / "training"
TRACKING_CONFIG = CONFIG_BASE_PATH / "tracking"


@hydra.main(
    version_base=None,
    config_path=TRAINING_CONFIG.as_posix(),
    config_name="config",
)
def run(config: DictConfig) -> None:
    pass


@hydra.main(
    version_base=None,
    config_path=TRACKING_CONFIG.as_posix(),
    config_name="config",
)
def run_tracker(config: DictConfig) -> None:
    logger.info("Tracking Started")
    print(config)