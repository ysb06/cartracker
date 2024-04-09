import logging
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

import cartracker_v2.utils as utils
from cartracker_v2.dataset.songdo_dataset import (
    SongdoDataset,
    yolovgg_test_collate_fn,
    yolovgg_train_collate_fn,
)
from cartracker_v2.models.yolovgg import Yolovgg


def get_dataloaders(config: DictConfig):
    dataset = SongdoDataset(**config["dataset"]["train"])
    training_dataset, validation_dataset = dataset.stratified_split()[0]
    test_dataset = SongdoDataset(**config["dataset"]["test"])

    training_loader = DataLoader(
        training_dataset,
        collate_fn=yolovgg_train_collate_fn,
        **config["training_loader"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        collate_fn=yolovgg_test_collate_fn,
        **config["validation_loader"],
    )
    test_loader = DataLoader(
        test_dataset, collate_fn=yolovgg_test_collate_fn, **config["test_loader"]
    )

    return training_loader, validation_loader, test_loader

def get_trainer(config: DictConfig, logger: str = "wandb"):
    device = utils.get_torch_device(config.device)

    run_name = f"CartrackerL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(name=run_name, project="SCK_Cartracker")
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    trainer = L.Trainer(
        **config.trainer,
        logger=wandb_logger,
        accelerator=device.type,
        profiler="advanced",
    )

    return trainer

def get_simple_prediction(model: Yolovgg, batch):
    return model.predict(batch[0])

def get_advanced_prediction(model: Yolovgg, batch):
    return model.predict(batch[0])


def run_simple_perdiction(output_name: str, model: Yolovgg, test_loader: DataLoader):
    fourcc = cv2.VideoWriter.fourcc(*"x264")
    video_writer = cv2.VideoWriter(
        f"./outputs/output_{output_name}.avi", fourcc, 29.97, (1280, 720)
    )
    if not video_writer.isOpened():
        print("File open failed!")
        test_loader.dataset.release()
        video_writer.release()
        return

    for batch in tqdm(test_loader):
        origs, plots, scigc_infos = get_advanced_prediction(model, batch)

        for idx, (plot, scigc_info) in enumerate(zip(plots, scigc_infos)):
            plot = utils.put_text(
                plot, f"Batch ID: {idx} / {len(plots) - 1}", (0, 40), (255, 0, 255)
            )
            if len(scigc_info) != 0:
                for xyxy, label in scigc_info:
                    plot = cv2.rectangle(
                        plot, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 5
                    )
                    plot = utils.put_text(
                        plot, f"Label: {label}", (xyxy[0], xyxy[1] - 20), (0, 0, 255)
                    )
                video_writer.write(plot)
                cv2.imshow("Plot", plot)
                cv2.waitKey(0)
            else:
                video_writer.write(plot)
                cv2.imshow("Plot", plot)
                cv2.waitKey(1)
    video_writer.release()
    test_loader.dataset.release()