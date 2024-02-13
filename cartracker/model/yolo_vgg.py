import logging
from typing import Any, Dict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import ultralytics
from albumentations.pytorch import ToTensorV2
from box import Box
from torchvision.models import VGG16_Weights, vgg16
from ultralytics.engine.results import Boxes, Results

logger = logging.getLogger(__name__)


class YoloTrackingModel(nn.Module):
    def __init__(self, config: Box) -> None:
        super().__init__()
        self.yolo_layer = ultralytics.YOLO(**config["yolo"])
        self.yolo_layer.train(data='coco128.yaml')
        self.vgg_layer = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_layer.classifier[6] = nn.Linear(
            self.vgg_layer.classifier[6].in_features, 2
        )
        self.yolo_layer.to(config["yolo"]["device"])
        self.vgg_layer.to(config["vgg"]["device"])

    def forward(self, x):
        pass


class YoloVggModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        print(config)
        self.yolo_model = ultralytics.YOLO(config["yolo_path"])
        self.vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_model.classifier[6] = nn.Linear(
            self.vgg_model.classifier[6].in_features, 2
        )

        chkpoint = torch.load(config["vgg_path"], map_location=torch.device("cpu"))
        self.vgg_model.load_state_dict(chkpoint["model_state"])

        self.yolo_model.to(torch.device("mps"))
        self.vgg_model.to(torch.device("mps"))

        self.vgg_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def predict(self, input_x):
        self.vgg_model.eval()

        yolo_results: Results = self.yolo_model.predict(source=input_x, conf=0.1)
        yolo_result = yolo_results[0]

        raw = yolo_result.orig_img
        scigc_list = []
        for box in yolo_result.boxes:
            box: Boxes
            if round(box.cls.item()) == 2:
                xyxy = box.xyxy.squeeze().round().int().tolist()
                target: np.ndarray = raw[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
                target = self.vgg_transform(image=target)

                input_x = target["image"].unsqueeze(0).to(torch.device("mps"))
                vgg_result: torch.Tensor = self.vgg_model(input_x)
                _, predicted = torch.max(vgg_result.data, 1)

                if predicted == 1:
                    scigc_list.append(xyxy)

        return raw, scigc_list
