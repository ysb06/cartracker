import logging
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

from .bagloader.v250805 import ScartBag

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def stitch(images: list[np.ndarray]) -> np.ndarray:
    # The stitcher internally performs feature detection, homography
    # estimation, seam finding, exposure compensation and multiband
    # blending. The ordering matters: left → mid → right.
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed (error code {status}).")
    return panorama


class ObjectDetector:
    def __init__(self):
        self.model = RFDETRBase()
        self.model.optimize_for_inference()

    def detect(self, image: np.ndarray):
        detections = self.model.predict(image, threshold=0.5)
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(
            annotated_image, detections, labels
        )
        return annotated_image


bag = ScartBag("./data/20250801_125812_test1_0.bag")
bag.print_bag_info()
data, total_count = bag.extract_camera_data()

object_detector = ObjectDetector()

for i in range(total_count):
    sample1 = data[list(data.keys())[1]][i]["image"]
    sample2 = data[list(data.keys())[2]][i]["image"]
    sample3 = data[list(data.keys())[3]][i]["image"]

    target_image: Optional[np.ndarray] = None
    try:
        target_image = stitch([sample1, sample2, sample3])
    except RuntimeError as e:
        logging.error(f"Stitching error: {e} at index {i}")
        target_image = sample2

    center_obj_result = object_detector.detect(sample2)
    target_obj_result = object_detector.detect(target_image)

    cv2.imshow("Center Object Detection", center_obj_result)
    cv2.imshow("Target Object Detection", target_obj_result)
    cv2.waitKey(0)
