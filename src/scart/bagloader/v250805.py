import collections
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import rosbag
from tqdm import tqdm

TopicTuple = collections.namedtuple(
    "TopicTuple", ["msg_type", "message_count", "connections", "frequency"]
)
CAMERA_TOPICS = [
    "/clpe_ros/cam_0/image_raw",
    "/clpe_ros/cam_1/image_raw",
    "/clpe_ros/cam_2/image_raw",
    "/clpe_ros/cam_3/image_raw",
]


class ScartBag:
    def __init__(self, path: str):
        self.bag = rosbag.Bag(path, "r")

    def print_bag_info(self):
        print("=== Bag Info ===")
        start_time = self.bag.get_start_time()
        end_time = self.bag.get_end_time()

        # Unix timestamp를 KST datetime 객체로 변환
        kst = ZoneInfo("Asia/Seoul")
        start_datetime = datetime.fromtimestamp(start_time, tz=kst)
        end_datetime = datetime.fromtimestamp(end_time, tz=kst)

        print(f"파일명: {self.bag.filename}")
        print(
            f"시작 시간 (KST): {start_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')} ({start_time})"
        )
        print(
            f"종료 시간 (KST): {end_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')} ({end_time})"
        )
        print(f"지속 시간: {end_time - start_time:.2f}초")
        print(f"전체 메시지 개수: {self.bag.get_message_count()}")
        print()
        print("=== Topics Info ===")
        topics_info: dict[str, TopicTuple] = self.bag.get_type_and_topic_info().topics

        for topic, topic_info in topics_info.items():
            print(f"Topic: {topic}")
            print(f"  Message Type: {topic_info.msg_type}")
            print(f"  Message Count: {topic_info.message_count}")
            print(f"  Connections: {topic_info.connections}")
            if topic_info.frequency is not None:
                print(f"  Frequency: {topic_info.frequency:.2f} Hz")
            else:
                print(f"  Frequency: N/A")
            print()

    def extract_camera_data(self, camera_topic=CAMERA_TOPICS):
        data = {topic: [] for topic in camera_topic}
        kst = ZoneInfo("Asia/Seoul")

        # 각 토픽별 메시지 개수를 미리 계산
        topics_info = self.bag.get_type_and_topic_info().topics
        total_messages = sum(
            topics_info[topic].message_count
            for topic in camera_topic
            if topic in topics_info
        )

        image: Optional[np.ndarray] = None
        with tqdm(total=total_messages, desc="Extracting...", unit="msg") as pbar:
            for topic, header, timestamp in self.bag.read_messages(topics=camera_topic):
                image = self._convert_image(header)
                data[topic].append(
                    {
                        "image": image,
                        "timestamp": datetime.fromtimestamp(timestamp.to_sec(), tz=kst),
                    }
                )

                # 진행상황 업데이트
                pbar.update(1)

        return data, total_messages

    def _convert_image(self, header):
        """ROS Image 메시지를 numpy 배열로 변환합니다."""
        if header.encoding == "bgr8":
            # BGR 8-bit 이미지
            image = np.frombuffer(header.data, dtype=np.uint8)
            image = image.reshape((header.height, header.width, 3))
        elif header.encoding == "rgb8":
            # RGB 8-bit 이미지
            image = np.frombuffer(header.data, dtype=np.uint8)
            image = image.reshape((header.height, header.width, 3))
            # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif header.encoding == "mono8":
            # 흑백 8-bit 이미지
            image = np.frombuffer(header.data, dtype=np.uint8)
            image = image.reshape((header.height, header.width))
        elif header.encoding == "yuv422":
            # YUV422 이미지 (UYVY 형식)
            image = np.frombuffer(header.data, dtype=np.uint8)
            image = image.reshape((header.height, header.width, 2))
            # YUV422를 BGR로 변환
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print("지원 가능한 인코딩: bgr8, rgb8, mono8, yuv422")
            raise ValueError(f"지원하지 않는 인코딩: {header.encoding}")

        return image

    def close(self):
        """bag 파일을 닫습니다."""
        if self.bag:
            self.bag.close()
            print("Bag file closed.")
        else:
            print("No bag file to close.")