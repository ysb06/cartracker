import os
from typing import Any, Dict

import cv2
import numpy as np
from torch.utils.data import DataLoader
from ultralytics import YOLO

from cartracker.dataset.yolov8_dataset import RegionOfInterest, SongdoDataset
from cartracker.trainer.yolo_trainer.trainer import YoloTrackingModel
from pytrainer import Worker


def execute(config: Dict[str, Any]):
    trainer = YoloTrainer(config)


class YoloTrainer(Worker):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.dataset = SongdoDataset(**self.config["dataset"]["dataset_params"])
        self.model = YoloTrackingModel(self.config["model"])

    def work(self) -> None:
        print("I Work!")





class DataPath:
    def __init__(self, path_raw: Dict[str, str]) -> None:
        self.raw_label_path = os.path.join(path_raw["root"], path_raw["label"])
        self.raw_video_path = os.path.join(path_raw["root"], path_raw["video"])


class YOLOv8_Trainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.dataset = SongdoDataset(**config["dataset"]["dataset_params"])

        vfourcc = cv2.VideoWriter.fourcc(*"mp4v")
        vwriter = cv2.VideoWriter("output_1.mp4", vfourcc, 29.975, (320, 240))

        # for idx, (rect, frame) in enumerate(self.dataset):
        #     frame: np.ndarray = frame
        #     rect: RegionOfInterest = rect
        #     # image = rect.get_croped_image()
        #     # rs_image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)

        #     cv2.rectangle(frame, **rect.get_cv_rect())
        #     frame = frame[100:340, 400:720]

        #     if idx > 120:
        #         vwriter.write(frame)

        #     cv2.imshow("Frame Show", frame)
        #     cv2.waitKey(1)
        #     print(idx)
        #     if idx == 400:
        #         break

        vwriter.release()
        self.dataset.release()
        cv2.destroyAllWindows()

        # 다 된 것으로 보임.
        self.dataloader = DataLoader(
            self.dataset, **config["dataset"]["dataloader_params"]
        )
        self.model = YOLO("yolov8.yaml")
        self.model.train(data="coco128.yaml", epochs=100, imgsz=640, device="mps")
        print("OK")
