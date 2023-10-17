import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def load_config(path: str):
    import yaml

    training_config = {}
    with open(path, "r") as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)
        logger.info(f"{path} loaded")

    return training_config
