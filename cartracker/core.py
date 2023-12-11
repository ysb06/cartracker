import importlib
import logging
import pprint

from cartracker.util import load_config

logger = logging.getLogger(__name__)


def main():
    config = load_config("./cartracker/config.yaml")
    mode_name = config["mode"]
    mode_config = config[mode_name]

    module_name = f"cartracker.{mode_name}.{mode_config['name']}"
    logger.info(f"Running [{module_name}] module")
    pprint.pprint(mode_config)

    module = importlib.import_module(module_name)
    module.execute(mode_config)


if __name__ == "__main__":
    main()
