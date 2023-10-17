import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from torch.utils.data import Dataset
import yaml

from cartracker.dataset.label_studio import (
    LSCFLabels,
    KeyFrameLabel,
    BoundingBox,
)
from cartracker.util import load_config


logger = logging.getLogger()


class Box:
    def __init__(
        self,
        point_1: Tuple[float, float],
        point_2: Optional[Tuple[float, float]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        cv_options: Dict[str, Any] = {"color": (0, 0, 255), "thickness": 1},
    ) -> None:
        self.x1, self.y1 = point_1
        self.x2, self.y2 = point_1
        if point_2 is not None:
            self.x2, self.y2 = point_2
        elif width is not None and height is not None:
            self.x2 += width
            self.y2 += height
        else:
            raise TypeError(
                "Pass arguments only with pair of points or width and height"
            )
        self.cv_rect_options = cv_options

    def get_cv_rect(self, width: Union[int, float], height: Union[int, float]):
        rect = {
            "pt1": (round(self.x1 / 100 * width), round(self.y1 / 100 * height)),
            "pt2": (round(self.x2 / 100 * width), round(self.y2 / 100 * height)),
            **self.cv_rect_options,
        }
        return rect

    def __repr__(self) -> str:
        return self.get_cv_rect(1, 1)


@dataclass
class KeyFrame:
    frame: int
    time: float
    enabled: bool
    x: float
    y: float
    width: float
    height: float
    rotation: float
    info: Optional[Dict[str, Any]] = None


@dataclass
class Frame:
    start_keyframe: KeyFrame
    end_keyframe: KeyFrame
    frame_number: int

    def __repr__(self) -> str:
        return f"Frame[{self.start_keyframe.info['label_name']}] ({self.frame_number}/{self.start_keyframe.info['frame_count']}) of Task[{self.task_id}]"

    @property
    def task_id(self) -> int:
        if self.start_keyframe.info["task_id"] != self.end_keyframe.info["task_id"]:
            logger.warning(
                f"The data is invalid: {self.start_keyframe.info['task_id']} != {self.end_keyframe['task_id']}"
            )
        return self.start_keyframe.info["task_id"]

    @property
    def total_frame(self) -> int:
        if (
            self.start_keyframe.info["frame_count"]
            != self.end_keyframe.info["frame_count"]
        ):
            logger.warning(
                f"The data is invalid: {self.start_keyframe.info['frame_count']} != {self.end_keyframe['frame_count']}"
            )
        return self.start_keyframe.info["frame_count"]

    @property
    def moving_ratio(self) -> float:
        return (self.frame_number - self.start_keyframe.frame) / (self.end_keyframe.frame - self.start_keyframe.frame)

    @property
    def label_box(self) -> Box:
        x1 = (
            self.start_keyframe.x
            + (self.end_keyframe.x - self.start_keyframe.x) * self.moving_ratio
        )
        y1 = (
            self.start_keyframe.y
            + (self.end_keyframe.y - self.start_keyframe.y) * self.moving_ratio
        )
        width = (
            self.start_keyframe.width
            + (self.end_keyframe.width - self.start_keyframe.width) * self.moving_ratio
        )
        height = (
            self.start_keyframe.height
            + (self.end_keyframe.height - self.start_keyframe.height)
            * self.moving_ratio
        )
        return Box((x1, y1), width=width, height=height)


class SongdoDataset(Dataset):
    def __init__(self, path: str, data_size: int = 0) -> None:
        self.dataset_path = path
        self.label_info = load_config(os.path.join(path, "info.yaml"))
        self.raw_label = LSCFLabels(
            os.path.join(path, self.label_info["label_file_name"])
        )
        self.keyframe_data = self.initialize_data()
        self.debug = True

        self.__current_task_id: int = -1
        self.__current_video_capture: Optional[cv2.VideoCapture] = None
        self.__current_video_width = 0
        self.__current_video_height = 0

        self.__cursor = -1

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

        return data

    def __len__(self):
        return len(self.keyframe_data)

    def __iter__(self):
        self.__cursor = -1
        return self

    def __next__(self) -> Tuple[Frame, Dict[str, Any], List[List[float]]]:
        self.__cursor += 1
        return self.__getitem__(self.__cursor)

    def __getitem__(
        self, index: int
    ) -> Tuple[Frame, Dict[str, Any], List[List[float]]]:
        item = self.keyframe_data[index]
        if item.task_id != self.__current_task_id:
            self.__current_task_id = item.task_id
            video_file_name = self.label_info["task_related_video_filename_list"][
                self.__current_task_id
            ]
            self.__current_video_capture = cv2.VideoCapture(
                os.path.join(self.dataset_path, video_file_name)
            )
            self.__current_video_width = self.__current_video_capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            self.__current_video_height = self.__current_video_capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )

        item_rect = item.label_box.get_cv_rect(
            self.__current_video_width, self.__current_video_height
        )

        self.__current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame_number)
        _, frame = self.__current_video_capture.read()

        return item, item_rect, frame


class Generator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.label_raw: Optional[List[Dict]] = None
        with open(config["data_path"]["label"]) as f:
            self.label_raw = json.load(f)
        self.video_file_folder_path = self.config["data_path"]["video"]
        self.video_width = 0
        self.video_height = 0

    def generate(self):
        logger.info("Generation Start")
        video_file_names = [file for file in os.listdir(self.video_file_folder_path)]

        if self.label_raw is not None:
            for idx, task in enumerate(self.label_raw):
                logger.info(
                    f"Processing Task [{task['id']} ({idx + 1}/{len(self.label_raw)})]"
                )
                self.process_task(task, video_file_names)

    def process_task(self, task: Dict, video_file_names: List[str]) -> bool:
        file_name: str = task["file_upload"].split("-")[1]
        if file_name not in video_file_names:
            logger.warning(f"{file_name} video not found. Skipping this task...")
            return

        file_path = os.path.join(self.video_file_folder_path, file_name)
        video_capture = cv2.VideoCapture(file_path)
        self.video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        for idx, annotation in enumerate(task["annotations"]):
            logger.info(
                f"Processing Annotation [{annotation['id']} ({idx + 1}/{len(task['annotations'])})]"
            )
            self.process_annotation(annotation, video_capture)

        # Task 완료
        video_capture.release()

    def process_annotation(self, annotation: Dict, video_capture: cv2.VideoCapture):
        for idx, result in enumerate(annotation["result"]):
            logger.info(
                f"Processing Result [{result['id']} ({idx + 1}/{len(annotation['result'])})]"
            )
            self.process_result(result, video_capture)

    def process_result(self, result: Dict, video_capture: cv2.VideoCapture):
        value = result["value"]

        label_frame_count = value["framesCount"]
        capture_frame_count = round(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture_fps = video_capture.get(cv2.CAP_PROP_FPS)
        label_mspf = round(value["duration"] / label_frame_count * 1000)
        capture_mspf = round(1 / capture_fps * 1000)

        if label_frame_count != capture_frame_count:
            logger.warning(
                f"The frame count in label({label_frame_count}) and video({capture_frame_count}) does not match."
            )

        if label_mspf != capture_mspf:
            logger.warning(
                f"The MSPF does not match (Label[{label_mspf}] <==> Video[{capture_mspf}])"
            )

        prev_frame: Optional[KeyFrameLabel] = None
        for idx, current_frame_raw in enumerate(value["sequence"]):
            current_frame = KeyFrameLabel(**current_frame_raw)
            current_frame.id = idx
            current_frame.capture = video_capture

            if prev_frame is not None:
                self.process_label(prev_frame, current_frame)

            prev_frame = current_frame

    def process_label(
        self, prev_frame: KeyFrameLabel, current_frame: KeyFrameLabel, mspf: int = 1
    ):
        if prev_frame.capture is not None and current_frame.capture is not None:
            video_capture = prev_frame.capture  # VideoCapture 무결성 확인 절차가 없음. 주의할 것.

            frame_count_in_label = current_frame.frame - prev_frame.frame
            for frame_idx, frame_num in enumerate(
                range(prev_frame.frame, current_frame.frame)
            ):
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = video_capture.read()  # 시간 오래 걸림
                rect = self.calculate_rect(
                    frame_idx, frame_count_in_label, prev_frame, current_frame
                )
                frame = self.process_frame(frame, rect)

                if ret:
                    cv2.imshow("Frame Show", frame)
                    if frame_idx == 0:
                        if cv2.waitKey(1) == 27:
                            break
                    else:
                        if cv2.waitKey(mspf) == 27:
                            break
                else:
                    logger.warning(f"No frame for number {frame_num}")
        else:
            raise Exception("VideoCapture is None")

    def process_frame(self, frame: np.ndarray, rect: BoundingBox):
        cv2.rectangle(frame, **rect.get_cv_rect())
        # 이미지 추출 및 레이블
        return frame

    def calculate_rect(
        self,
        current_frame_idx: int,
        total_frame_count: int,
        label_1: KeyFrameLabel,
        label_2: KeyFrameLabel,
    ) -> BoundingBox:
        ratio = current_frame_idx / total_frame_count
        x1 = label_1.x + (label_2.x - label_1.x) * ratio
        y1 = label_1.y + (label_2.y - label_1.y) * ratio
        width = label_1.width + (label_2.width - label_1.width) * ratio
        height = label_1.height + (label_2.height - label_1.height) * ratio
        return BoundingBox(
            x1 * self.video_width / 100,
            y1 * self.video_height / 100,
            width=width * self.video_width / 100,
            height=height * self.video_height / 100,
        )

    def convert_annotations(self, annotation_results: List[Dict[str, Any]]):
        pass
