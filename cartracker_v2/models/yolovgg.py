from typing import Any, Dict, List, Optional, Tuple, Union
import os
import logging

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import ultralytics
from albumentations.pytorch import ToTensorV2
from box import Box
from torch import nn
from torch.optim import Optimizer
from torchvision.models import VGG16_Weights, vgg16
from ultralytics.engine.results import Boxes, Results

CAR_LABEL = 2

TensorInput = Union[torch.Tensor, List[torch.Tensor]]
IntInput = Union[int, List[int]]
YoloOutput = Union[Results, List[Results]]
InputBatch = Tuple[TensorInput, TensorInput, TensorInput, IntInput]

logger = logging.getLogger()


class VggLayer(nn.Module):
    def __init__(self, checkpoint: Optional[Dict[str, str]] = None, out_features: int = 2) -> None:
        super().__init__()
        best_path = checkpoint["best"] if checkpoint else None
        self.backbone_layer = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone_layer.classifier[6] = nn.Linear(
            self.backbone_layer.classifier[6].in_features, out_features
        )
        if os.path.exists(best_path):
            states = torch.load(best_path, map_location=torch.device("cpu"))
            try:
                self.backbone_layer.load_state_dict(states["model_state"])
                logger.info(f"Checkpoint {best_path} loaded")
            except KeyError:
                logger.warning("Warning: Checkpoint is not loaded")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone_layer(x)
        return x


class YoloModule:
    def __init__(
        self,
        yolo_config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.yolo_layer = ultralytics.YOLO(**yolo_config["args"])
        self.yolo_config = yolo_config

    def predict(self, x_input: TensorInput, *args, **kwargs) -> List[Results]:
        yolo_results: Union[Results, List[Results]] = self.yolo_layer.predict(
            x_input, *args, **kwargs
        )
        if type(yolo_results) == Results:
            yolo_results = [yolo_results]

        return yolo_results

    def train(self):
        if self.yolo_config["need_training"]:
            logger.info("Training YOLO Model...")
            self.yolo_layer.train(**self.yolo_config["training_args"])
            logger.info("Training Complete")
        else:
            logger.info("YOLO Training Skipped")


class Yolovgg(L.LightningModule):
    def __init__(
        self,
        yolo_config: Dict[str, Any],
        vgg_config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.yolo = YoloModule(yolo_config)
        self.vgg_layer = VggLayer(**vgg_config)
        self.vgg_config = vgg_config

        self.vgg_loss = nn.CrossEntropyLoss()
        self.optimizer_args = None
        # self.scheduler_args = None

        self.vgg_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def predict(
        self, x_input: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[Tuple[List[int], int]]]]:
        yolo_results: List[Results] = self.yolo.predict(
            x_input, conf=0.5, verbose=False
        )

        origs: List[np.ndarray] = []
        plots: List[np.ndarray] = []
        scigc_car_info: List[List[List]] = []

        for idx_1, yolo_result in enumerate(yolo_results):
            orig: np.ndarray = yolo_result.orig_img
            origs.append(orig)
            plots.append(yolo_result.plot())

            car_xyxy_images: List[List] = []
            for box in yolo_result.boxes:
                box: Boxes = box
                xyxy: List[int] = box.xyxy.squeeze().round().int().tolist()
                target: np.ndarray = orig[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]

                if box.cls == CAR_LABEL:
                    car_xyxy_images.append(
                        [xyxy, self.vgg_transform(image=target)["image"]]
                    )
            scigc_car_info.append(car_xyxy_images)

        car_images = []
        cursors = []
        for idx_1, info in enumerate(scigc_car_info):
            for idx_2, (xyxy, image) in enumerate(info):
                car_images.append(image)
                cursors.append((idx_1, idx_2))

        if len(car_images) != 0:
            vgg_output: torch.Tensor = self.vgg_layer(torch.stack(car_images))

            for cursor, output in zip(cursors, vgg_output):
                # scigc_car_info[cursor[0]][cursor[1]].append(torch.argmax(output).item())
                scigc_car_info[cursor[0]][cursor[1]] = (
                    scigc_car_info[cursor[0]][cursor[1]][0],
                    torch.argmax(output).item(),
                )

        return origs, plots, scigc_car_info

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.AdamW(self.vgg_layer.parameters(), lr=1e-3)
        return optimizer

    def on_train_start(self) -> None:
        self.yolo.train()

    def training_step(self, batch: InputBatch, _):
        # batch is come from DataLoader with collate_fn in songdo_dataset.py
        # Todo: Yolo 모델을 1280x720 크기로 다시 학습 시키기, 해당 이미지 크기를 받을 수 있어야 한다.
        # yolo_x: torch.Tensor = batch[0]
        # roi: torch.Tensor = batch[1]
        vgg_xs: torch.Tensor = batch[2]
        true_labels: torch.Tensor = batch[3]

        # YOLO 학습이 구현되어 있지 않음..Todo list

        # 현재는 VGG만 학습
        logits = self.vgg_layer(vgg_xs)
        loss = self.vgg_loss(logits, true_labels)

        self.log("train_loss", loss, on_epoch=False, on_step=True)

        acc = (torch.argmax(logits, dim=1) == true_labels).float().mean().item()

        return loss

    def on_train_epoch_end(self) -> None:
        torch.save(self.vgg_layer.state_dict(), self.vgg_config["checkpoint"]["last"])

    def validation_step(self, batch, _):
        vgg_xs: torch.Tensor = batch[2]
        true_labels: torch.Tensor = batch[3]

        # YOLO 학습이 구현되어 있지 않음..Todo list

        # 현재는 VGG만 학습
        logits = self.vgg_layer(vgg_xs)
        loss = self.vgg_loss(logits, true_labels)

        self.log("valid_loss", loss, on_epoch=False, on_step=True)

        acc = (torch.argmax(logits, dim=1) == true_labels).float().mean().item()

        return loss

    def on_validation_epoch_end(self) -> None:
        torch.save(self.vgg_layer.state_dict(), self.vgg_config["checkpoint"]["best"])
