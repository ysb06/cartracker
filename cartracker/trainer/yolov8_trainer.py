from typing import Any, Dict
from ultralytics import YOLO
import os
from cartracker.dataset.yolov8_dataset import SongdoDataset
from cartracker.util import load_config
from torch.utils.data import DataLoader
import cv2


def execute(config: Dict[str, Any]):
    trainer = YOLOv8_Trainer(config)


class DataPath:
    def __init__(self, path_raw: Dict[str, str]) -> None:
        self.raw_label_path = os.path.join(path_raw["root"], path_raw["label"])
        self.raw_video_path = os.path.join(path_raw["root"], path_raw["video"])


class YOLOv8_Trainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.dataset = SongdoDataset(**config["dataset"]["dataset_params"])
        for idx, (item, rect, frame) in enumerate(self.dataset):
            cv2.rectangle(frame, **rect)
            cv2.imshow("Frame Show", frame)
            cv2.waitKey(1)
        # 다 된 것으로 보임.
        self.dataloader = DataLoader(self.dataset, **config["dataset"]["dataloader_params"])
        print('OK')
