import logging
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import cartracker_v2.utils as utils
from cartracker_v2.dataset.songdo_dataset import (
    SongdoDataset,
    yolovgg_test_collate_fn,
    yolovgg_train_collate_fn,
)
from cartracker_v2.models.yolovgg import Yolovgg
from cartracker_v2.core import get_dataloaders, get_trainer, run_simple_perdiction

logger = logging.getLogger()


@hydra.main(
    version_base=None,
    config_path=(Path(__file__).parent / ".." / "config").as_posix(),
    config_name="config",
)
def run(config: DictConfig) -> None:
    L.seed_everything(config.seed)
    training_loader, validation_loader, test_loader = get_dataloaders(config)
    model = Yolovgg(**config["model"])
    trainer = get_trainer(config)

    # trainer.fit(
    #     model,
    #     train_dataloaders=training_loader,
    #     val_dataloaders=validation_loader,
    # )
    training_loader.dataset.dataset.release()
    logger.info("Training Complete")

    print("Predicting Model...")
    run_simple_perdiction(trainer.logger.name, model, test_loader)
    print("Prediction Complete")


if __name__ == "__main__":
    run()
