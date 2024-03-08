from typing import Tuple
import torch
import numpy as np
import cv2


def get_torch_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        supported_devices = ["cuda", "mps", "cpu"]
        if device in supported_devices:
            return torch.device(device)
        else:
            raise ValueError(f"Unsupported device: {device}")


def put_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
