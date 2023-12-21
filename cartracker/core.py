import importlib
import logging
import pprint
from typing import Any, Dict, List

from cartracker.util import load_config

logger = logging.getLogger(__name__)


def main(config_path: str = "./cartracker/config.yaml"):
    config = load_config(config_path)
    mode_name: str = config["mode"]
    
    mode_config: Dict[str, Any] = config
    for key in mode_name.split("."):
        mode_config = mode_config[key]
    module_name = f"cartracker.{mode_name}"
    logger.info(f"Running [{module_name}] module")
    pprint.pprint(mode_config)

    module = importlib.import_module(module_name)
    module.execute(mode_config)


if __name__ == "__main__":
    main()
