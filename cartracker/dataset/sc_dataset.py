from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset, Subset, DataLoader
import os
import cv2
import torch
import albumentations as A
from sklearn.model_selection import train_test_split
import random
from albumentations.pytorch import ToTensorV2


class SCDataset(Dataset):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.path: str = config["path"]
        self.classes: List[str] = sorted(config["classes"])
        self.data_list: List[Tuple[str, int]] = []

        for class_id, class_name in enumerate(self.classes):
            full_path = os.path.join("./", self.path, class_name)
            for file_path in sorted(os.listdir(full_path)):
                self.data_list.append((os.path.join(full_path, file_path), class_id))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) -> Tuple[Any, int]:
        file_path, class_id = self.data_list[index]
        raw = cv2.imread(file_path)
        # if self.transform:
        #     augmented = self.transform(image=raw)
        #     raw = augmented["image"]
        # img = raw.astype(float) / 255.0
        # img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        return raw, class_id

    def __get_class_indices(self):
        class_indices = {class_id: [] for class_id in range(len(self.classes))}
        for idx, (_, class_id) in enumerate(self.data_list):
            class_indices[class_id].append(idx)
        return class_indices

    def split(self, train_frac: float, test_frac: float):
        # 나중에 random_state 추가
        class_indices = self.__get_class_indices()
        train_indices, val_indices, test_indices = [], [], []

        for indices in class_indices.values():
            train_idx, temp_idx = train_test_split(indices, test_size=1 - train_frac)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_frac / (1 - train_frac)
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)

        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )


def get_dataloaders(
    dataset: SCDataset,
    training_frac: float,
    test_frac: float,
    training_config: Dict[str, Any] = {},
    training_transform: A.Compose = None,
):
    training_set, validation_set, test_set = dataset.split(training_frac, test_frac)

    training_loader = DataLoader(
        training_set, collate_fn=SCCollate(training_transform), **training_config
    )
    validation_loader = DataLoader(validation_set, collate_fn=SCCollate())
    test_loader = DataLoader(test_set, collate_fn=SCCollate())

    return training_loader, validation_loader, test_loader


class SCCollate:
    def __init__(
        self,
        transform=A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    ) -> None:
        self.transform = transform

    def __call__(self, batch) -> Any:
        images, labels = [], []
        for img, label in batch:
            # albumentations transform 적용
            augmented = self.transform(image=img)
            img = augmented["image"]
            images.append(img)
            labels.append(label)

        # 이미지 배치와 레이블 배치 생성
        images = torch.stack(images)
        labels = torch.tensor(labels)

        return images, labels
