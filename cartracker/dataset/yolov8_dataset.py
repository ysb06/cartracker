import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
from torch.utils.data import Dataset
import numpy as np

from cartracker.dataset.label_studio import Frame, KeyFrame, LSCFLabels, BoundingBox
from cartracker.util import load_config

logger = logging.getLogger()


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

    def get_croped_image(self):
        return self.raw_image[
            self.point_1[1] : self.point_2[1], self.point_1[0] : self.point_2[0]
        ]


class VideoLoader:
    def __init__(
        self,
        root_path: str,
        filenames_with_id: Dict[int, str],
        fixed_video_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.root_path = root_path
        self.filename_dict = filenames_with_id
        self.current_id = -1
        self.current_video_capture: Optional[cv2.VideoCapture] = None
        self.__video_size = None if fixed_video_size is None else fixed_video_size

    @property
    def video_size(self):
        # 무조건 get_frame 이후에 불러와야 한다는 점이 문제. 본 기능을 get_frame에 포함하든지, 다른 설계방안을 찾을 것.
        if self.__video_size is not None:
            return self.__video_size
        elif self.current_video_capture is not None:
            width = self.current_video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.current_video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return (width, height)
        else:
            logger.warning("Nothing is specified for this video loader")
            return None

    def get_frame(self, id: int, frame_number: int):
        if id != self.current_id:
            self.current_id = id
            video_file_name = self.filename_dict[id]
            video_file_path = os.path.join(self.root_path, video_file_name)
            self.current_video_capture = cv2.VideoCapture(video_file_path)

        self.current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = self.current_video_capture.read()

        return frame
    
    def release(self):
        self.current_video_capture.release()


class SongdoDataset(Dataset):
    def __init__(self, path: str, data_size: int = 0) -> None:
        self.dataset_path = path
        self.label_info = load_config(os.path.join(path, "info.yaml"))
        label_file_path = os.path.join(path, self.label_info["label_file_name"])
        self.raw_label = LSCFLabels(label_file_path)
        self.video_loader = VideoLoader(
            path,
            self.label_info["task_related_video_filename_list"],
        )

        self.keyframe_data = self.initialize_data()

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

    def __len__(self):
        return len(self.keyframe_data)

    def __getitem__(self, index: int) -> Tuple[RegionOfInterest, np.ndarray]:
        item = self.keyframe_data[index]
        frame = self.video_loader.get_frame(item.task_id, item.frame_number)
        item_rect = RegionOfInterest(frame, item.label_box)
        # 주의: get_cv_rect는 get_frame후에 불려져야 함.

        return item_rect, frame, item.label_name
    
    def release(self):
        self.video_loader.release()
