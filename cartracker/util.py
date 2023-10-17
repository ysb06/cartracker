import logging
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str):
    config = {}
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        logger.info(f"{path} loaded")

    return config
