import torch

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