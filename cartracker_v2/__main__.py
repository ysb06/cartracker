from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

import cartracker_v2.utils as utils
from cartracker_v2.dataset.songdo_dataset import (
    SongdoDataset,
    yolovgg_collate_fn,
    yolovgg_test_collate_fn,
    yolovgg_train_collate_fn,
)
from cartracker_v2.models.yolovgg import Yolovgg
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path=(Path(__file__).parent / ".." / "config").as_posix(),
    config_name="config",
)
def run(config: DictConfig) -> None:
    device = utils.get_torch_device(config.device)
    L.seed_everything(config.seed)

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

    model = Yolovgg(**config["model"])

    # wandb_logger = WandbLogger(name="Cartracker Lightning", project="STGCN_WAVE")
    # wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    # trainer = L.Trainer(
    #     **config.trainer,
    #     logger=False,
    #     accelerator=device.type,
    # )
    # trainer.fit(
    #     model,
    #     train_dataloaders=training_loader,
    #     val_dataloaders=validation_loader,
    # )

    # trainer.predict(model, test_loader)
    print("Predicting Model...")
    for batch in tqdm(test_loader):
        origs, plots, scigc_infos = model.predict(*batch)

        for idx, (plot, scigc_info) in enumerate(zip(plots, scigc_infos)):
            plot = cv2.putText(
                plot,
                f"Batch ID: {idx} / {len(plots) - 1}",
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 255),
                5,
            )
            if len(scigc_info) != 0:
                for xyxy, car_image, label in scigc_info:
                    plot = cv2.rectangle(
                        plot, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 5
                    )
                    plot = cv2.putText(
                        plot,
                        f"Label: {label}",
                        (xyxy[0], xyxy[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        5,
                    )
                cv2.imshow("Plot", plot)
                cv2.waitKey(0)
            else:
                cv2.imshow("Plot", plot)
                cv2.waitKey(1)


if __name__ == "__main__":
    run()
