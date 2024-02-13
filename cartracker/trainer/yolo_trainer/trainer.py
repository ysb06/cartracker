from pytrainer import Worker
from cartracker.dataset.yolov8_dataset import SongdoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import wandb
from ...model.yolo_vgg import YoloTrackingModel

logger = logging.getLogger(__name__)


class YoloTrainer(Worker):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        logger.info(f"Starting {self.config['wandb']['project']} Project...")
        # wandb.init(config=self.config, **self.config["wandb"])

        self.dataset = SongdoDataset(**self.config["dataset"])
        training_dataset, validation_dataset = self.dataset.stratified_split()[0]
        self.training_loader = DataLoader(
            training_dataset, **self.config["training_loader"]
        )
        self.validation_loader = DataLoader(
            validation_dataset, **self.config["validation_loader"]
        )
        self.model = YoloTrackingModel(self.config["model"])

    def work(self) -> None:
        logger.info("Start Training...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.train(self.training_loader)
            self.validate(self.validation_loader)
    
    def train(self, loader: DataLoader) -> None:
        pass

    def validate(self, loader: DataLoader) -> None:
        pass

    def finish_work(self) -> None:
        wandb.finish()
        self.dataset.release()
        logger.info("Training finished.")
