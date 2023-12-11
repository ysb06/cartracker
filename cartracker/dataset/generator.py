from typing import Any, Dict, Union

from cartracker.dataset.yolov8_dataset import RegionOfInterest, SongdoDataset
import os
from ultralytics import YOLO
import numpy as np
import cv2
import tqdm

DATASET_PATH = "./datasets/sc_images/"


def execute(config: Dict[str, Any]):
    Generator(config)


class Generator:
    def __init__(self, config: Dict[str, Union[str, int, float, bool]]) -> None:
        self.dataset = SongdoDataset(config["dataset"]["path"])

        count = 0
        for _, (rect, frame, label) in tqdm.tqdm(enumerate(self.dataset), total=len(self.dataset)):
            frame: np.ndarray = frame
            rect: RegionOfInterest = rect
            label: str = label

            sc_image = rect.get_croped_image()
            if label == "SCIGC_Vehicle":
                cv2.imwrite(os.path.join(DATASET_PATH, f"{count}.png"), sc_image)
                count += 1
        
        print(f"Total Saved Image: {count}")
