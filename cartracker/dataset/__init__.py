from typing import Any, Dict
from .yolov8_dataset import Generator


def generate_yolov8_dataset(config: Dict[str, Any]):
    generator = Generator(config)
    generator.generate()
