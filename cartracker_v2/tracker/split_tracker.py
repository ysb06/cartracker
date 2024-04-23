import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from cartracker_v2.dataset.songdo_dataset import SongdoDataset
from cartracker_v2.recorder.video_recorder import Recorder
from cartracker_v2.tracker.deep_sort.iou_matching import iou

logger = logging.getLogger(__name__)


class SplittedImage:
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

    def boxes(self):
        for cls, xywh, xyxy in zip(
            self.prediction.boxes.cls,
            self.prediction.boxes.xywh,
            self.prediction.boxes.xyxy,
        ):
            n_xywh = [xywh[0] + self.pt1[0], xywh[1] + self.pt1[1], xywh[2], xywh[3]]
            n_xyxy = [
                xyxy[0] + self.pt1[0],
                xyxy[1] + self.pt1[1],
                xyxy[2] + self.pt1[0],
                xyxy[3] + self.pt1[1],
            ]

            yield cls, torch.Tensor(n_xywh), torch.Tensor(n_xyxy)


class SceneImage:
    def __init__(
        self,
        image: np.ndarray,
        crop_w_size: int = 640,
        crop_h_size: int = 640,
    ) -> None:
        crop_result = self.__crop(image, crop_w_size, crop_h_size)

        self.image = image
        self.cropped_image_matrix: List[List[SplittedImage]] = crop_result[
            "cropped_image_matrix"
        ]
        self.cropped_image_list: List[SplittedImage] = crop_result["cropped_image_list"]
        self.count: Tuple[int, int] = crop_result["count"]
        self.interval: Tuple[int, int] = crop_result["interval"]
        self.intersected_image_dict: Dict[SplittedImage, List[SplittedImage]] = (
            self.__get_intersected_image_dict(self.cropped_image_matrix)
        )
        self.cropped_image_width = crop_w_size
        self.cropped_image_height = crop_h_size

        self.prediction: Optional[Results] = None

    def __get_intersected_image_dict(self, image_matrix: List[List[SplittedImage]]):
        result = {}
        for i in range(len(image_matrix)):
            for j in range(len(image_matrix[i])):
                intersected_images = []
                if i > 0:
                    intersected_images.append(image_matrix[i - 1][j])
                    if j > 0:
                        intersected_images.append(image_matrix[i - 1][j - 1])
                    if j < len(image_matrix[i]) - 1:
                        intersected_images.append(image_matrix[i - 1][j + 1])
                if i < len(image_matrix) - 1:
                    intersected_images.append(image_matrix[i + 1][j])
                    if j > 0:
                        intersected_images.append(image_matrix[i + 1][j - 1])
                    if j < len(image_matrix[i]) - 1:
                        intersected_images.append(image_matrix[i + 1][j + 1])
                if j > 0:
                    intersected_images.append(image_matrix[i][j - 1])
                if j < len(image_matrix[i]) - 1:
                    intersected_images.append(image_matrix[i][j + 1])

                result[image_matrix[i][j]] = intersected_images
        return result

    def __crop(self, image: np.ndarray, crop_w_size, crop_h_size):
        horizontal_count = math.ceil(image.shape[1] / crop_w_size) + 1
        vertical_count = math.ceil(image.shape[0] / crop_h_size) + 1

        horizontal_interval = round(
            (image.shape[1] - crop_w_size) / (horizontal_count - 1)
        )
        vertical_interval = round((image.shape[0] - crop_h_size) / (vertical_count - 1))

        cropped_image_matrix: List[List[SplittedImage]] = [
            [] for _ in range(vertical_count)
        ]
        cropped_image_list: List[SplittedImage] = []
        for i in range(vertical_count):
            for j in range(horizontal_count):
                cropped_image = SplittedImage(
                    image,
                    j,
                    i,
                    horizontal_interval,
                    vertical_interval,
                    crop_w_size,
                    crop_h_size,
                )
                cropped_image_matrix[i].append(cropped_image)
                cropped_image_list.append(cropped_image)

        result = {
            "cropped_image_matrix": cropped_image_matrix,
            "cropped_image_list": cropped_image_list,
            "count": (horizontal_count, vertical_count),
            "interval": (horizontal_interval, vertical_interval),
        }
        return result

    def detect(
        self,
        model: YOLO,
        conf: float = 0.5,
        verbose: bool = False,
    ):
        image_raws = [img_obj.image for img_obj in self.cropped_image_list]

        predictions: List[Results] = model.predict(
            image_raws,
            conf=conf,
            verbose=verbose,
            imgsz=(self.cropped_image_height, self.cropped_image_width),
        )
        for idx, result in enumerate(predictions):
            self.cropped_image_list[idx].prediction = result

        self.prediction = predictions

    def combine_predictions(self, label: int = 2):  # label 2: car
        # 어떤 박스 내에 같은 레이블을 가진 박스가 없다고 가정
        temp_boxes: List[Tuple[torch.Tensor, torch.Tensor]] = []
        all_xyxy = []

        for current_image in self.cropped_image_list:
            image_list = [current_image] + self.intersected_image_dict[current_image]
            for image in image_list:
                for cls, xywh, xyxy in image.boxes():
                    if cls == label:
                        # xyxy, xywh에 원래 이미지 포인트로 바꾸는 작업 필요 Todo
                        all_xyxy.append(xyxy)

                        max_iou_box: Tuple[torch.Tensor, torch.Tensor] = None
                        max_iou_val = -1
                        max_iou_box_idx = -1
                        for idx, box in enumerate(temp_boxes):
                            iou_val = iou(box[0], xywh.unsqueeze(dim=0))
                            if iou_val > 0 and iou_val > max_iou_val:
                                max_iou_val = iou_val
                                max_iou_box = box
                                max_iou_box_idx = idx

                        if max_iou_box is not None:
                            temp_boxes[max_iou_box_idx] = self.__combine_bbox(
                                max_iou_box[1], xyxy
                            )
                        else:
                            temp_boxes.append((xywh, xyxy))

        return temp_boxes, all_xyxy

    def __combine_bbox(self, bbox1, bbox2):
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])

        xywh = torch.Tensor((x1, y1, x2 - x1, y2 - y1))
        xyxy = torch.Tensor((x1, y1, x2, y2))

        return (xywh, xyxy)


class Tracker:
    def __init__(self) -> None:
        self.yolo_model = YOLO(model="./models/yolov8n_960.pt")
        self.train_dataset = SongdoDataset(f"./datasets/sc_videos/dataset/train")
        self.test_dataset = SongdoDataset(f"./datasets/sc_videos/dataset/test")

    def run(self) -> None:
        video_recorder = Recorder(size=(3840, 2160))

        with tqdm(total=len(self.train_dataset)) as pbar:
            for idx, (roi, img, label) in enumerate(self.test_dataset):
                if img.shape[0] != 2160 or img.shape[1] != 3840:
                    break

                if idx > 60:
                    break

                if label == "SCIGC" or label == "Unknown":
                    target_image = SceneImage(img, crop_w_size=3840, crop_h_size=2160)
                    target_image.detect(self.yolo_model)
                    result, all_xyxy = target_image.combine_predictions()

                    frame = self.__combine_result_plot(img, result, all_xyxy)
                    cv2.waitKey(0)
                    self.__test_prediction_plot(target_image)

                    video_recorder.write(frame)

                pbar.update(1)

        video_recorder.close()

    def __combine_result_plot(self, image: np.ndarray, result, all_xyxy):
        canvas = image.copy()
        for xyxy in all_xyxy:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(
                img=canvas,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=(0, 255, 0),
                thickness=3,
            )

        for _, xyxy in result:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(
                img=canvas,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=(0, 0, 255),
                thickness=3,
            )

        cv2.imshow("Combined Result", canvas)

        return canvas

    def __test_intersection_plot(self, image: SceneImage):
        for img_obj in image.cropped_image_list:
            canvas = np.zeros_like(image.image)
            for img in image.intersected_image_dict[img_obj]:
                canvas[img.pt1[1] : img.pt2[1], img.pt1[0] : img.pt2[0]] = img.image
            cv2.imshow("test", canvas)

        return canvas

    def __test_prediction_plot(self, image: SceneImage):
        canvas = np.zeros_like(image.image)
        for sp_img in image.cropped_image_list:
            canvas[sp_img.pt1[1] : sp_img.pt2[1], sp_img.pt1[0] : sp_img.pt2[0]] = (
                sp_img.prediction.plot()
            )
            cv2.imshow("Prediction Test", canvas)
            cv2.waitKey(0)

        return canvas
