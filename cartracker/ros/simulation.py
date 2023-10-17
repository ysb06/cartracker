import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import rosbag
import tqdm

logger = logging.getLogger()

# 필요한 경우 코드 다시 작성할 것


class Simulation:
    def __init__(self, data_root_path: str) -> None:
        self.raw_paths = [
            os.path.join(data_root_path, item)
            for item in os.listdir(data_root_path)
            if os.path.isdir(os.path.join(data_root_path, item))
        ]
        self.callbacks: List[
            Callable[[pd.Timestamp, float, float], Optional[Tuple[str, Any]]]
        ] = []

    def add_callback(
        self,
        callback: Callable[[pd.Timestamp, float, float], Optional[Tuple[str, Any]]],
    ):
        self.callbacks.append(callback)

    def simulate(self):
        result: Dict[str, pd.DataFrame] = {}
        for path_item in self.raw_paths:
            logger.info(f"Simulating {path_item}...")

            data = pd.DataFrame()
            bag_file_paths = sorted(os.listdir(path_item))
            for idx, file_name in enumerate(bag_file_paths):
                if os.path.splitext(file_name)[-1] != ".bag":
                    logger.info(f"Passing file which is not bag file: {file_name}")
                    continue

                bag_file_path = os.path.join(path_item, file_name)
                logger.info(
                    f"[{idx + 1}/{len(bag_file_paths)}] Loading {bag_file_path}..."
                )
                bag_raw = rosbag.Bag(bag_file_path)
                logger.info("Loading Complete")  # Rosbag 파일 로딩 완료

                bag_data: Dict[str, List] = {
                    key: []
                    for key in ["Timestamp", "Datetime", "Latitude", "Longitude"]
                }
                for _, message, ros_time in tqdm.tqdm(
                    bag_raw.read_messages(topics=["/gps/gps"])
                ):
                    bag_data["Timestamp"].append(ros_time.to_nsec())
                    bag_data["Datetime"].append(
                        pd.Timestamp(ros_time.to_nsec(), tz="Asia/Seoul")
                    )
                    bag_data["Latitude"].append(message.latitude)
                    bag_data["Longitude"].append(message.longitude)

                    for callback in self.callbacks:
                        label, value = callback(
                            bag_data["Datetime"][-1],
                            bag_data["Latitude"][-1],
                            bag_data["Longitude"][-1],
                        )
                        if label not in bag_data:
                            bag_data[label] = [
                                None for _ in range(len(bag_data["Timestamp"]))
                            ]
                        else:
                            bag_data[label].append(None)

                        bag_data[label][-1] = value

                bag_result = pd.DataFrame(bag_data)
                data = pd.concat([data, bag_result])

            result[path_item] = data

        return result
