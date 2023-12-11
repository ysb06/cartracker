from typing import Any, Dict
import ultralytics
import torch.nn as nn

def execute(config: Dict[str, Any]):
    print("OK")
    print(config)

class Yolo8Vgg(nn.Module):
    def __init__(self) -> None:
        model = ultralytics.YOLO()