import collections
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger()


class BoundingBox:
    def __init__(
        self,
        point_1: Tuple[float, float],
        point_2: Optional[Tuple[float, float]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        cv_options: Dict[str, Any] = {},
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
        return (self.frame_number - self.start_keyframe.frame) / (
            self.end_keyframe.frame - self.start_keyframe.frame
        )

    @property
    def label_box(self) -> BoundingBox:
        ratio = self.moving_ratio

        x_move = (self.end_keyframe.x - self.start_keyframe.x) * ratio
        y_move = (self.end_keyframe.y - self.start_keyframe.y) * ratio
        width_change = (self.end_keyframe.width - self.start_keyframe.width) * ratio
        height_change = (self.end_keyframe.height - self.start_keyframe.height) * ratio

        x1 = self.start_keyframe.x + x_move
        y1 = self.start_keyframe.y + y_move
        width = self.start_keyframe.width + width_change
        height = self.start_keyframe.height + height_change

        return BoundingBox((x1, y1), width=width, height=height)
    
    @property
    def label_name(self) -> str:
        return self.start_keyframe.info['label_name']


@dataclass
class LabelElement:
    data: List[Dict[str, Any]]
    data_info: List[Dict[str, Any]]
    index: List[int]
    depth: int
    position: int


class NestedIterator:
    def __init__(
        self, raw: List[Dict[str, Any]], keys: List[Union[str, List[str]]]
    ) -> None:
        self.raw = raw
        self.keys = keys
        self.stack = collections.deque(
            [LabelElement(self.raw, [], [], 0, 0)]
        )  # Additional depth for tracking current depth
        self.current_item = None

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], List[int]]]:
        return self

    def __next__(self) -> Tuple[Dict[str, Any], List[int]]:
        while self.stack:
            element = self.stack[-1]
            data_list = element.data
            infos = element.data_info
            if element.position >= len(data_list):
                self.stack.pop()
                continue

            current_item = data_list[element.position]
            current_indices = element.index + [element.position]

            # Update the current position for the next iteration
            element.position += 1

            # If we're at the desired depth, return the item
            if element.depth == len(self.keys):
                return current_item, infos, current_indices

            check_key = self.keys[element.depth]
            if type(check_key) is not str:
                check_key = check_key[0]

            # Otherwise, dive deeper if possible
            if element.depth < len(self.keys) and check_key in current_item:
                next_key = self.keys[element.depth]
                next_item = current_item
                if type(next_key) is not str:
                    for key in next_key:
                        if key in next_item:
                            next_item = next_item[key]
                else:
                    next_item = current_item[next_key]

                info = [item for item in infos]
                info.append(current_item)
                self.stack.append(
                    LabelElement(next_item, info, current_indices, element.depth + 1, 0)
                )

        raise StopIteration


class LSCFLabels:
    def __init__(self, path: str) -> None:
        self.raw: Optional[List[Dict]] = None
        self.keys = ["annotations", "result", ["value", "sequence"]]
        with open(path) as file:
            self.raw = json.load(file)
        self.iterator = NestedIterator(self.raw, self.keys)
        self.label_info = None

    def __iter__(self) -> NestedIterator:
        self.iterator = NestedIterator(self.raw, self.keys)
        return self

    def __next__(self):
        item, infos, indices = self.iterator.__next__()
        return item, infos, indices
