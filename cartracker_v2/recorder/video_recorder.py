from typing import Optional, Tuple
import cv2
import numpy as np
import os
import datetime


class Recorder:
    def __init__(
        self,
        output_folder: str = "./outputs",
        output_file: Optional[str] = None,
        video_format: str = "mp4v",
        size: Tuple[int, int] = (640, 480),
        frame_rate: float = 29.975,
    ) -> None:
        vfourcc = cv2.VideoWriter.fourcc(*video_format)
        if output_file is None:
            output_file = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".mp4"
        output_path = os.path.join(output_folder, output_file)
        self.writer = cv2.VideoWriter(output_path, vfourcc, frame_rate, size)

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def close(self):
        self.writer.release()
