from typing import Any, Dict
import logging
from cartracker.model.yolo_vgg import YoloVggModel
import cv2


logger = logging.getLogger(__name__)


def execute(config: Dict[str, Any]):
    model = YoloVggModel(config=config["model"])

    test_config = config["test"]

    capture = cv2.VideoCapture(test_config["target_path"])
    frame_count = 0
    while capture.isOpened():
        frame_count += 1
        logger.info(frame_count)
        _, frame = capture.read()

        raw, results = model.predict(frame)  # save predictions as labels

        if len(results) > 0:
            for xyxy in results:
                raw = cv2.rectangle(raw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 3)
        else:
            continue
        
        cv2.imshow("Result", raw)
        cv2.waitKey(0)

        

