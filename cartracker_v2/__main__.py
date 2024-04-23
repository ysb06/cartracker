import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

import importlib


logger = logging.getLogger()

CONFIG_BASE_PATH = Path(__file__).parent / ".." / "config"


@hydra.main(
    version_base=None,
    config_path=CONFIG_BASE_PATH.as_posix(),
    config_name="config",
)
def run(config: DictConfig) -> None:
    if "tracking" in config:
        module = importlib.import_module(
            f".{config.tracking.name}", package="cartracker_v2.tracker"
        )
        tracker = module.Tracker(config.tracking)
        tracker.run()


run()
