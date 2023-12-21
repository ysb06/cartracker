import logging
from typing import Any, Dict, Generator

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ultralytics
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights
from tqdm import tqdm
from ultralytics.engine.results import Boxes, Results
from albumentations.pytorch import ToTensorV2


from cartracker.dataset.sc_dataset import SCDataset, get_dataloaders
from cartracker.util import seed

YOLO_MODEL_PATH = "./models/y8v.pt"

logger = logging.getLogger(__name__)


def execute(config: Dict[str, Any]):
    logger.info("Executing...")
    yolo_training_config = config["yolo"]["training"]
    if yolo_training_config["activated"]:
        trainer = YoloTrainer(yolo_training_config)
        trainer.train()

    yolo_test_config = config["yolo"]["test"]
    if yolo_test_config["activated"]:
        tester = YoloTester(config=yolo_test_config)
        tester.run()

    vgg_training_config = config["vgg"]["training"]
    if vgg_training_config["activated"]:
        trainer = VggTrainer(vgg_training_config)
        trainer.train()


class VggTrainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        seed(config["seed"])
        self.epochs = config["epochs"]
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, len(config["dataset"]["classes"])
        )
        # Pretrained Model Path: /Users/sbyim/.cache/torch/hub/checkpoints/vgg16-397923af.pth
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), **config["optimizer"]["params"]
        )

        self.train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        self.dataset = SCDataset(config["dataset"])
        (
            self.training_dataloader,
            self.validation_loader,
            self.test_loader,
        ) = get_dataloaders(
            self.dataset,
            0.7,
            0.2,
            training_config=config["dataloader"]["params"],
            training_transform=self.train_transform,
        )
        # wandb.init(project=f"SCIGC YV Model VGG Training", config=config)

    def train(self):
        logger.info("Training...")

        min_val_loss = np.inf
        for epoch in range(self.epochs):
            logger.info(f"Epoch: {epoch + 1} / {self.epochs}")
            self.model.train()
            learning_loss = 0.0
            for i, data in tqdm(
                enumerate(self.training_dataloader), total=len(self.training_dataloader)
            ):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss: torch.Tensor = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                learning_loss += loss.item()
                if i % 2000 == 1999:
                    print(f"[{epoch + 1}, {i + 1}] loss: {learning_loss / 2000:.3f}")
                    learning_loss = 0.0
                # wandb.log({"Learning Loss": learning_loss})
            val_loss, _ = self.validate()
            if val_loss < min_val_loss:
                state = {"model_state": self.model.state_dict()}
                torch.save(state, f"./models/best_vgg.pt")

        self.test()

    def validate(self):
        self.model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 기울기 계산 비활성화
            for data in self.validation_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.validation_loader)
        val_accuracy = 100 * correct / total
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        return val_loss, val_accuracy

    def test(self):
        self.model.eval()  # 모델을 평가 모드로 설정
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 기울기 계산 비활성화
            for data in self.test_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(self.test_loader)
        test_accuracy = 100 * correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        return test_loss, test_accuracy


class YoloTester:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.yolo_model = ultralytics.YOLO(config["model_path"])
        self.test_params = config["params"]

    def run(self):
        results: Generator[Results] = self.yolo_model(**self.test_params, stream=True)
        logger.info(f"Test Complete")

        count = 0
        for idx, result in enumerate(results):
            # 반복마다 예측을 수행하는 것은 비효율적이므로 cv에서 읽어 특정 프레임마다 처리
            if idx % 100 != 0:
                continue
            raw = result.orig_img
            for box in result.boxes:
                box: Boxes = box
                xyxy = box.xyxy.squeeze().round().int().tolist()
                target: np.ndarray = raw[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
                # cv2.imshow(f"Label: {result.names[box.cls.item()]}", target)
                # cv2.waitKey(1)
                if round(box.cls.item()) == 2:
                    cv2.imwrite(f"./datasets/sc_images/NR_SCIGC/{count}.png", target)
                    count += 1
            if count % 100 == 0:
                logger.info(f"NR Counts: {count}")

            if count >= 1203:
                break


class YoloTrainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.yolo_model = ultralytics.YOLO(config["model_path"])
        self.training_params = config["params"]

    def train(self) -> None:
        logger.info("Training Model...")
        self.yolo_model.train(**self.training_params)
        logger.info("Exporting Model...")
        self.yolo_model.export()


class Yolo8Vgg(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.yolo = ultralytics.YOLO("./model/yolov8n.pt")

    def forward(self, input) -> None:
        self.yolo(input)
