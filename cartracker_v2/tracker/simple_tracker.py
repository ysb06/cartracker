from typing import Any, Dict, List, Optional, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
import matplotlib.pyplot as plt
import cv2
import math
import logging
from cartracker_v2.dataset.songdo_dataset import SongdoDataset
import numpy as np

logger = logging.getLogger(__name__)


class SplitedImage:
    def __init__(
        self,
        original_image: np.ndarray,
        x: int,
        y: int,
        x_interval: int,
        y_interval: int,
        width: int,
        height: int,
    ) -> None:
        self.original_image = original_image
        self.position = (x, y)
        self.pt1 = (x * x_interval, y * y_interval)
        self.pt2 = (self.pt1[0] + width, self.pt1[1] + height)
        self.image = original_image[
            self.pt1[1] : self.pt2[1], self.pt1[0] : self.pt2[0]
        ]
        self.prediction: Optional[Results] = None

    def plot_location(self, color=(0, 0, 255), thickness=3):
        result = cv2.rectangle(
            img=self.original_image.copy(),
            pt1=self.pt1,
            pt2=self.pt2,
            color=color,
            thickness=thickness,
        )
        return result
    
    @property
    def plot_prediction(self):
        return self.prediction.plot()


class Tracker:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.yolo_model = YOLO(**config["yolo"])
        self.train_dataset = SongdoDataset(f"{config['dataset']['path']}/train")
        self.test_dataset = SongdoDataset(f"{config['dataset']['path']}/test")

    def __crop_image(
        self, img: np.ndarray, model_imgsz_h: int = 640, model_imgsz_w: int = 640
    ):
        vertical_count = math.ceil(img.shape[0] / model_imgsz_h) + 1
        horizontal_count = math.ceil(img.shape[1] / model_imgsz_w) + 1

        vp_interval = round((img.shape[0] - model_imgsz_h) / (vertical_count - 1))
        hp_interval = round((img.shape[1] - model_imgsz_w) / (horizontal_count - 1))

        cropped_images: Dict[Tuple[int, int], SplitedImage] = {}
        for i in range(vertical_count):
            for j in range(horizontal_count):
                cropped_img = SplitedImage(
                    img, j, i, hp_interval, vp_interval, model_imgsz_w, model_imgsz_h
                )
                cropped_images[cropped_img.position] = cropped_img
        return cropped_images

    def run(self) -> None:
        for roi, img, label in self.test_dataset:
            if label == "SCIGC" or label == "Unknown":
                cropped_list = self.__crop_image(img)
                positions = list(cropped_list.keys())
                cropped_images = list(cropped_list.values())

                predictions: List[Results] = self.yolo_model.predict([cimg.image for cimg in cropped_images], conf=0.5)
                for idx, result in enumerate(predictions):
                    position = positions[idx]
                    cropped_list[position].prediction = result
            
            for cimg in cropped_images:
                cv2.imshow("Target", cimg.plot_location())
                cv2.imshow("Prediction",cimg.plot_prediction)
                cv2.waitKey(0)
                # 이제 각 cropped image에 대한 prediction을 합치는 방안을 연구
