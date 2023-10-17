import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import collections
import cv2


class BoundingBox:
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: float = 1,
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x1
        self.y2 = y1
        if x2 is not None and y2 is not None:
            self.x2 = x2
            self.y2 = y2
        elif width is not None and height is not None:
            self.x2 += width
            self.y2 += height

        self.color = color
        self.thickness = thickness

    def get_cv_rect(self):
        return {
            "pt1": (round(self.x1), round(self.y1)),
            "pt2": (round(self.x2), round(self.y2)),
            "color": self.color,
            "thickness": self.thickness,
        }


@dataclass
class KeyFrameLabel:
    frame: int
    time: float
    enabled: bool
    x: float
    y: float
    width: float
    height: float
    rotation: float
    id: int = -1
    capture: Optional[cv2.VideoCapture] = None


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
            if (
                element.depth < len(self.keys)
                and check_key in current_item
            ):
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
                    LabelElement(
                        next_item,
                        info,
                        current_indices,
                        element.depth + 1,
                        0
                    )
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

