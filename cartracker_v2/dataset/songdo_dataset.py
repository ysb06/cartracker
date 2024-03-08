import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Subset

from cartracker_v2.dataset.label_studio import BoundingBox, Frame, KeyFrame, LSCFLabels

logger = logging.getLogger()


def load_config(path: str):
    config = {}
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        logger.info(f"{path} loaded")

    return config


class RegionOfInterest:
    def __init__(
        self,
        raw_image: np.ndarray,
        bounding_box: BoundingBox,
        cv_options: Dict[str, Any] = {"color": (0, 0, 255), "thickness": 2},
    ) -> None:
        self.raw_image = raw_image
        self.bounding_box = bounding_box
        bounding_box.cv_rect_options = cv_options

        self.point_1 = (
            round(bounding_box.x1 / 100 * self.raw_image.shape[1]),
            round(bounding_box.y1 / 100 * self.raw_image.shape[0]),
        )
        self.point_2 = (
            round(bounding_box.x2 / 100 * self.raw_image.shape[1]),
            round(bounding_box.y2 / 100 * self.raw_image.shape[0]),
        )

    def get_cv_rect(self):
        rect = {
            "pt1": self.point_1,
            "pt2": self.point_2,
            **self.bounding_box.cv_rect_options,
        }
        return rect

    def get_cropped_image(self):
        yyxx = self.yyxx
        return self.raw_image[yyxx[0] : yyxx[1], yyxx[2] : yyxx[3]]

    @property
    def yyxx(self):
        return self.point_1[1], self.point_2[1], self.point_1[0], self.point_2[0]


class VideoLoader:
    def __init__(self, root_path: str, filenames_with_id: Dict[int, str]) -> None:
        self.root_path = root_path
        self.filename_dict = filenames_with_id

        self.video_captures: Dict[int, cv2.VideoCapture] = {}

    def get_frame(self, task_id: int, frame_number: int):
        if task_id not in self.video_captures:
            video_file_path = os.path.join(self.root_path, self.filename_dict[task_id])
            self.video_captures[task_id] = cv2.VideoCapture(video_file_path)

        self.video_captures[task_id].set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = self.video_captures[task_id].read()

        return frame

    def release(self):
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()


class SongdoDataset(Dataset):
    def __init__(self, path: str, for_test: bool = False) -> None:
        self.dataset_path = path
        self.label_info = load_config(os.path.join(path, "info.yaml"))
        label_file_path = os.path.join(path, self.label_info["label_file_name"])
        self.raw_label = LSCFLabels(label_file_path)
        self.video_loader = VideoLoader(path, self.label_info["video_filename_list"])

        self.frame_data = self.initialize_data()

    def initialize_data(self) -> List[Frame]:
        data = []

        integrity_check_set = set()
        prev_keyframe = None
        prev_keyframe_num = -1
        current_info = {}
        for res, info, label_idx in self.raw_label:
            check_key = tuple(label_idx[:3])
            if check_key not in integrity_check_set:
                # 새로운 Label, Annotation, Video로 시작
                current_info = {
                    "task_id": info[0]["id"],
                    "label_name": info[2]["value"]["labels"][0],
                    "frame_count": info[2]["value"]["framesCount"],
                }
                integrity_check_set.add(check_key)

                current_keyframe = KeyFrame(**res, info=current_info)
                prev_keyframe = current_keyframe
                prev_keyframe_num = label_idx[3]
                # 귀찮아서지만 의도적으로 마지막 키프레임은 처리하지 않았음.
            else:
                if label_idx[3] != prev_keyframe_num + 1:
                    logger.warning("Raw labels are not arranged")

                current_keyframe = KeyFrame(**res, info=current_info)

                for frame_number in range(prev_keyframe.frame, current_keyframe.frame):
                    frame = Frame(prev_keyframe, current_keyframe, frame_number)
                    data.append(frame)

                prev_keyframe = current_keyframe
                prev_keyframe_num = label_idx[3]
        # 어떻게 효율적으로 무결성 있게 작성할지는 고민해 보자
        return data

    def stratified_split(
        self,
        n_splits=5,
        shuffle=True,
        random_state=42,
        # split_level: Literal['frame', 'keyframe', 'task'] = 'frame'   # 추후 필요하면 개발
    ) -> List[Tuple[Subset, Subset]]:
        # Test 셋은 별도로 준비할 것

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        # frame level
        labels = [frame.label_name for frame in self.frame_data]
        subsets = [
            (Subset(self, train_idx), Subset(self, val_idx))
            for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels)
        ]

        return subsets

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, index: int) -> Tuple[RegionOfInterest, np.ndarray, str]:
        item = self.frame_data[index]
        frame = self.video_loader.get_frame(item.task_id, item.frame_number)
        item_rect = RegionOfInterest(frame, item.label_box)
        # 주의: get_cv_rect는 get_frame후에 불려져야 함.

        return item_rect, frame, item.label_name

    def release(self):
        self.video_loader.release()


# 차후에 collate_fn이 여러개 생길 경우 별도로 모듈을 만들어서 관리할 필요가 있음
vgg_train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def yolovgg_collate_fn(batch: List[Tuple[RegionOfInterest, np.ndarray, str]]) -> Tuple:
    frames = torch.stack([torch.Tensor(item[1]) for item in batch]).permute(
        0, 3, 1, 2
    )  # YOLO 학습용
    rois = torch.stack([torch.Tensor(item[0].yyxx) for item in batch])  # YOLO 학습용
    # YOLO Label은 모두 Car로 동일 (Label Number: 2)
    cropped_images = torch.stack(
        [torch.Tensor(item[0].get_cropped_image()) for item in batch]
    )  # VGG 학습용
    labels = torch.stack(
        [0 if item[2] == "SCIGC" else 1 for item in batch]
    )  # VGG 학습용
    # SCIGC: 0
    # NR_SCIGC: 1

    return frames, rois, cropped_images, labels


yolo_test_transform = A.Compose(
    [
        # A.Resize(640, 640),
        # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ]
)
vgg_test_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def yolovgg_train_collate_fn(
    batch: List[Tuple[RegionOfInterest, np.ndarray, str]]
) -> Tuple:
    frames = [item[1] for item in batch]
    rois = [item[0].yyxx for item in batch]  # YOLO 학습용
    cropped_images = torch.stack(
        [
            vgg_train_transform(image=item[0].get_cropped_image())["image"]
            for item in batch
        ]
    )  # VGG 학습용
    labels = torch.tensor(
        [0 if item[2] == "SCIGC" else 1 for item in batch]
    )  # VGG 학습용
    return frames, rois, cropped_images, labels


def yolovgg_test_collate_fn(
    batch: List[Tuple[RegionOfInterest, np.ndarray, str]]
) -> Tuple:
    frames = [item[1] for item in batch]
    rois = [item[0].yyxx for item in batch]  # YOLO 학습용
    cropped_images = torch.stack(
        [
            vgg_test_transform(image=item[0].get_cropped_image())["image"]
            for item in batch
        ]
    )  # VGG 학습용
    labels = torch.tensor(
        [0 if item[2] == "SCIGC" else 1 for item in batch]
    )  # VGG 학습용
    return frames, rois, cropped_images, labels
