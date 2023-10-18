from typing import Any, Dict
from ultralytics import YOLO
import os
from cartracker.dataset.label_studio import BoundingBox
from cartracker.dataset.yolov8_dataset import RegionOfInterest, SongdoDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np


def execute(config: Dict[str, Any]):
    trainer = YOLOv8_Trainer(config)


class DataPath:
    def __init__(self, path_raw: Dict[str, str]) -> None:
        self.raw_label_path = os.path.join(path_raw["root"], path_raw["label"])
        self.raw_video_path = os.path.join(path_raw["root"], path_raw["video"])


class YOLOv8_Trainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.dataset = SongdoDataset(**config["dataset"]["dataset_params"])
        for idx, (rect, frame) in enumerate(self.dataset):
            frame: np.ndarray = frame
            rect: RegionOfInterest = rect
            image = rect.get_croped_image()
            rs_image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)

            # cv2.rectangle(frame, **rect.get_cv_rect())
            cv2.imshow("Frame Show", rs_image)
            cv2.waitKey(1)
        # 다 된 것으로 보임.
        self.dataloader = DataLoader(self.dataset, **config["dataset"]["dataloader_params"])
        self.model = YOLO()
        print('OK')
