import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import time
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
import pandas as pd

from cartracker_v2.dataset.songdo_dataset import SongdoDataset, Frame
from cartracker_v2.models.yolovgg import VggLayer

logger = logging.getLogger(__name__)


def convert_tensor(tensor: Tensor) -> np.ndarray:
    # 텐서를 넘파이 배열로 변환 (값 범위 0-1)
    img: np.ndarray = tensor.numpy()
    img = img.transpose(1, 2, 0)  # CHW to HWC

    # 정규화 해제
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # 역정규화

    # 이미지 값 범위를 0-255로 변환
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


class VggLayer(nn.Module):
    def __init__(
        self, checkpoint_path: str = "./models/vgg16.pt", out_features: int = 2
    ) -> None:
        super().__init__()
        self.backbone_layer = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone_layer.classifier[6] = nn.Linear(
            self.backbone_layer.classifier[6].in_features, out_features
        )
        if os.path.exists(checkpoint_path):
            states = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            try:
                self.backbone_layer.load_state_dict(states["model_state"])
                logger.info(f"Checkpoint {checkpoint_path} loaded")
            except KeyError:
                logger.warning("Warning: Checkpoint is not loaded")

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone_layer(x)
        return x


@dataclass
class Prediction:
    yolo_result: Results
    yolo_class: Tuple[int, str]
    xyxy: Tensor  # 첫 xy는 좌상단, 두번째 xy는 우하단
    xywh: Tensor  # 첫 xy는 중앙점
    vgg_class: Tuple[int, str] = (-1, "Unknown")

    @property
    def original_image(self) -> np.ndarray:
        return self.yolo_result.orig_img

    @property
    def image(self) -> np.ndarray:
        xyxy = self.xyxy.int().tolist()
        image = self.original_image[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        return image

    def plot(self, original_image: Optional[np.ndarray] = None):
        if original_image is None:
            original_image = self.original_image

        xyxy = self.xyxy.int().tolist()
        color = (0, 0, 255) if self.vgg_class[0] == 0 else (255, 0, 0)

        cv2.rectangle(original_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
        cv2.putText(
            original_image,
            self.vgg_class[1],
            (xyxy[0], xyxy[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        return original_image


class SimpleModel:
    def __init__(self, target_label: int = 2) -> None:
        self.yolo_model = YOLO(model="./models/yolov8n_960.pt")
        self.vgg_model = VggLayer(checkpoint_path="./models/vgg16.pt")
        self.vgg_model = self.vgg_model.to("mps")
        self.yolo_model
        self.target_label = target_label
        self.vgg_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        self.last_prediction_unit_time = -1
        self.last_prediction_time = -1

    def __predict(
        self,
        images: List[np.ndarray],
        conf: float = 0.5,
        imgsz: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Dict[Results, List[Prediction]], List[Results]]:
        start_time = time.time()
        input_length = len(images)

        imgsz: Tuple[int, int] = imgsz if imgsz is not None else images[0].shape[:2]
        yolo_results: List[Results] = self.yolo_model.predict(
            images, conf=conf, imgsz=imgsz, verbose=False, device="mps"
        )

        result: List[Prediction] = []
        result_images: List[np.ndarray] = []
        result_dict: Dict[Results, List[Prediction]] = {}

        for yolo_result in yolo_results:
            result_dict[yolo_result] = []

            for y_cls, y_xyxy, y_xywh in zip(
                yolo_result.boxes.cls,
                yolo_result.boxes.xyxy,
                yolo_result.boxes.xywh,
            ):
                if y_cls == self.target_label:
                    prediction = Prediction(yolo_result, y_cls, y_xyxy, y_xywh)
                    result.append(prediction)
                    result_images.append(prediction.image)
                    result_dict[yolo_result].append(prediction)

        result_images: List[np.ndarray] = [
            self.vgg_transform(image=img)["image"] for img in result_images
        ]
        result_images: Tensor = torch.stack(result_images)
        result_images = result_images.to("mps")
        vgg_results: Tensor = self.vgg_model(result_images)
        for idx, vgg_result in enumerate(vgg_results):
            label = vgg_result.argmax().item()
            result[idx].vgg_class = (label, "SCIGC" if label == 0 else "Non-SCIGC")

        # Performance Measurement
        self.last_prediction_unit_time = (time.time() - start_time) / input_length
        self.last_prediction_time = time.time() - start_time

        return result_dict, yolo_results

    def predict(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        conf: float = 0.5,
        imgsz: Optional[Tuple[int, int]] = None,
    ) -> List[List[Prediction]]:
        if isinstance(images, np.ndarray):
            images = [images]

        predict_target_list: List[List[np.ndarray]] = []
        last_shape = None
        for image in images:
            if image.shape != last_shape:
                last_shape = image.shape
                predict_target_list.append([])
            predict_target_list[-1].append(image)

        predictions: Dict[Results, List[Prediction]] = {}
        yolo_results: List[Results] = []
        for predict_target in predict_target_list:
            prediction, yolo_result = self.__predict(
                predict_target, conf=conf, imgsz=imgsz
            )
            predictions.update(prediction)
            yolo_results.extend(yolo_result)

        return [
            predictions[res] for res in yolo_results
        ]  # 주의: 장면 별로 결과가 분리되어 있지 않음. Prediction.yolo_result를 사용해서 구분해야 함


class Tracker:
    def __init__(self) -> None:
        self.model = SimpleModel()
        self.train_dataset = SongdoDataset(f"./datasets/sc_videos/dataset/train")
        self.test_dataset = SongdoDataset(f"./datasets/sc_videos/dataset/test")
        self.batch_size = 16
        self.result_dict = {
            "frame_num": [],
            "recording_time": [],
            "label_code": [],
            "label_name": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
            "c1": [],
            "c2": [],
            "width": [],
            "height": [],
            "original_shape": [],
        }

    def close(self) -> None:
        self.train_dataset.release()
        self.test_dataset.release()

    def plot(self) -> None:
        with tqdm(total=len(self.test_dataset) // self.batch_size + 1) as pbar:
            pbar.desc = "Test"
            batch = []
            for idx, (roi, image, label) in enumerate(self.test_dataset):
                batch.append(image)
                if len(batch) >= self.batch_size:
                    batch_results = self.model.predict(batch, imgsz=(2176, 3840))
                    # 주의 반환 형식 변경됨, 코드 다시 작성 필요
                    logger.info(
                        f"Prediction Time: {self.model.last_prediction_time:.4f} sec"
                    )
                    logger.info(
                        f"Prediction Unit Time: {self.model.last_prediction_unit_time:.4f} sec"
                    )

                    prev_scene = batch_results[0].yolo_result
                    prev_plot = None
                    for result in batch_results:
                        if prev_scene != result.yolo_result:
                            cv2.imshow("test", prev_plot)
                            cv2.waitKey(0)
                            prev_scene = result.yolo_result

                        prev_plot = result.plot()

                    batch = []
                    pbar.update(1)

    def __record(
        self,
        frame_number: int,
        prediction: Prediction,
    ) -> None:
        xyxy_box = prediction.xyxy
        xywh_box = prediction.xywh
        label_code = prediction.vgg_class[0]
        label_name = prediction.vgg_class[1]

        self.result_dict["frame_num"].append(frame_number)
        self.result_dict["label_code"].append(label_code)
        self.result_dict["label_name"].append(label_name)
        self.result_dict["x1"].append(xyxy_box[0].item())
        self.result_dict["y1"].append(xyxy_box[1].item())
        self.result_dict["x2"].append(xyxy_box[2].item())
        self.result_dict["y2"].append(xyxy_box[3].item())
        self.result_dict["c1"].append(xywh_box[0].item())
        self.result_dict["c2"].append(xywh_box[1].item())
        self.result_dict["width"].append(xywh_box[2].item())
        self.result_dict["height"].append(xywh_box[3].item())
        self.result_dict["original_shape"].append(prediction.original_image.shape)

    def run(
        self,
        batch_size: int = 4,
        start_time: str = "2023-08-01T08:49:05.000000Z",
        end_time: str = "2023-08-01T08:49:29.000000Z",
    ) -> None:
        start_time = start_time.replace("Z", "+00:00")
        start_date_time = datetime.fromisoformat(start_time)
        end_time = end_time.replace("Z", "+00:00")
        end_date_time = datetime.fromisoformat(end_time)

        frame_batch: List[Tuple[np.ndarray, Frame]] = []
        count = 0
        with tqdm(total=len(self.test_dataset), leave=False) as pbar:
            idx = 0
            while idx < len(self.test_dataset):
                frame_info = self.test_dataset.frame_data[idx]

                if frame_info.task_id == 2:  # 2번 SAMPLE 비디오에 대해서만 추출
                    if idx > 1362 and idx < 1794:
                        frame_mat = self.test_dataset.video_loader.get_frame(
                            frame_info.task_id, frame_info.frame_number
                        )
                        frame_batch.append((frame_mat, frame_info))
                        pbar.set_description(
                            f"Batching ({len(frame_batch)} / {batch_size})..."
                        )
                    else:
                        pbar.set_description("Skipping...")

                    if (
                        len(frame_batch) >= batch_size
                        or idx >= len(self.test_dataset) - 1
                    ):
                        count += 1
                        pbar.set_description(f"Predicting ({len(frame_batch)})...")
                        image_batch = [frame for frame, _ in frame_batch]
                        info_batch = [info for _, info in frame_batch]
                        if count == 3:
                            pass
                        batch_results = self.model.predict(image_batch)

                        for box_results, info in zip(batch_results, info_batch):
                            frame_number = info.frame_number
                            plot_canvas = None
                            if len(box_results) <= 0:
                                logger.warning(
                                    f"If NMS time limit is exceeded, the result may be empty. (Frame: {frame_number})"
                                )
                                continue
                            plot_canvas = box_results[0].original_image.copy()
                            for result in box_results:
                                self.__record(frame_number, result)
                                plot_canvas = result.plot(original_image=plot_canvas)
                            cv2.imwrite(
                                f"./outputs/sample_images/{frame_number:03}.png",
                                plot_canvas,
                            )

                        frame_batch = []

                    # if key_input >= 49 and key_input <= 57:
                    #     skip_count = (key_input - 48) * 10
                    #     idx += skip_count - 1
                    #     pbar.update(skip_count - 1)

                pbar.update()
                idx += 1

        # for frame_mat, _ in frame_batch:
        #     cv2.imshow("Test", frame_mat)
        #     cv2.waitKey(1)
        self.result_dict["recording_time"] = self.__calc_datetime(
            self.result_dict["frame_num"], start_date_time, end_date_time
        )
        import pandas as pd

        result_data = pd.DataFrame(self.result_dict)
        result_data.to_excel("./outputs/result_data.xlsx", index=False)

    def __calc_datetime(
        self, frame_numbers: List[int], start_datetime: datetime, end_datetime: datetime
    ) -> List[datetime]:
        frame_number_count = max(frame_numbers) - min(frame_numbers)
        frame_timedelta = (end_datetime - start_datetime) / frame_number_count

        return [
            (start_datetime + frame_timedelta * n).strftime("%F %T.%f")
            for n in range(frame_number_count)
        ]


# 기존 데이터셋 로딩 속도가 굉장히 느림. 파일 크기가 클수록 더 느림. 물론 Dataloader를 사용하면 병렬로 처리해서 빠를 수는 있음.


def recalc_time(
    data: pd.DataFrame,
    start_time: str = "2023-08-01T08:49:05.000000Z",
    end_time: str = "2023-08-01T08:49:29.000000Z",
):
    start_time = start_time.replace("Z", "+00:00")
    start_datetime = datetime.fromisoformat(start_time)
    end_time = end_time.replace("Z", "+00:00")
    end_datetime = datetime.fromisoformat(end_time)
    frame_col: List[int] = data["frame_num"].to_list()

    min_frame_number = min(frame_col)
    frame_number_count: int = max(frame_col) - min_frame_number
    frame_timedelta = (end_datetime - start_datetime) / frame_number_count

    datetime_col = [
        (start_datetime + frame_timedelta * (frame_num - min_frame_number)).strftime("%F %T.%f")
        for frame_num in frame_col
    ]

    data["recording_time"] = pd.Series(datetime_col)

    return data
