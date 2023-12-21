import logging
import yaml
import random
import numpy as np
import torch
from torch import device
import platform
import os

logger = logging.getLogger(__name__)


def load_config(path: str):
    config = {}
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        logger.info(f"{path} loaded")

    return config


def seed(seed, cuda_available: bool = False, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device_automatically():
    result = (
        device("cuda")
        if torch.cuda.is_available()
        else device("mps")
        if torch.backends.mps.is_available()
        else device("cpu")
    )
    logger.info(f"Device is automatically select as {result}")
    return result
