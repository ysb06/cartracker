from typing import Any, Dict, List, Tuple, Union

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


class Yolovgg(L.LightningModule):
    def __init__(self, yolo_config: Dict[str, Any], vgg_config: Dict[str, Any]) -> None:
        super().__init__()
        self.yolo_layer = ultralytics.YOLO(**yolo_config)
        # self.yolo_layer.train(data="coco128.yaml")
        self.vgg_layer = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_layer.classifier[6] = nn.Linear(
            self.vgg_layer.classifier[6].in_features, vgg_config["out_features"]
        )

        #
        if vgg_config["checkpoint"] is not None:
            chkpoint = torch.load(
                vgg_config["checkpoint"], map_location=torch.device("cpu")
            )
            self.vgg_layer.load_state_dict(chkpoint["model_state"])

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

    def forward(
        self,
        x_input: Union[torch.Tensor, List[torch.Tensor]],
        roi: Union[torch.Tensor, List[torch.Tensor]],
        cropped_input: Union[torch.Tensor, List[torch.Tensor]],
        label: Union[int, List[int]],
    ) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray], torch.Tensor]:
        yolo_results: Union[Results, List[Results]] = self.yolo_layer.predict(
            x_input, conf=0.1, show_labels=False, show_conf=False
        )
        if type(yolo_results) == Results:
            yolo_results = [yolo_results]

        origs: List[np.ndarray] = []
        xyxys: List[List[int]] = []
        car_images: List[torch.Tensor] = []
        for yolo_result in yolo_results:
            orig: np.ndarray = yolo_result.orig_img
            for box in yolo_result.boxes:
                box: Boxes = box
                xyxy: List[int] = box.xyxy.squeeze().round().int().tolist()
                target: np.ndarray = orig[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
                if box.cls == CAR_LABEL:
                    origs.append(orig)
                    xyxys.append(xyxy)
                    car_images.append(self.vgg_transform(image=target)["image"])

        vgg_output: torch.Tensor = self.vgg_layer(torch.stack(car_images))

        return origs, xyxys, vgg_output

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.AdamW(self.vgg_layer.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: InputBatch, _):
        # batch is come from DataLoader with collate_fn in songdo_dataset.py

        # Todo: Yolo 모델을 1280x720 크기로 다시 학습 시키기, 해당 이미지 크기를 받을 수 있어야 한다.
        yolo_x: TensorInput = batch[0]
        roi: TensorInput = batch[1]
        vgg_x: TensorInput = batch[2]
        label: IntInput = batch[3]

        yolo_results: YoloOutput = self.yolo_layer.predict(yolo_x, conf=0.1)
        # YOLO 학습이 구현되어 있지 않음..Todo list

        # 현재는 VGG만 학습
        vgg_result = self.vgg_layer(vgg_x)
        loss = self.vgg_loss(vgg_result, label)

        return loss

    def on_train_epoch_end(self) -> None:
        return

    def validation_step(self, batch, _):
        return

    def on_validation_epoch_end(self) -> None:
        return

    def predict_step(self, batch, _) -> Any:
        return
