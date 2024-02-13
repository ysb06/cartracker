import importlib
import inspect
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional, Union

import yaml
from box import Box


@dataclass
class PyTrainerArgsDef:
    module: str
    config: str


PyTrainerArgs = Union[Namespace, PyTrainerArgsDef]


class Worker:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)

    def work(self) -> None:
        raise NotImplementedError("No work for this worker")
    
    def finish_work(self) -> None:
        pass


def execute(args: PyTrainerArgs):
    worker_class = get_worker_class(args.module)
    worker: Worker = worker_class(args.config)
    worker.work()
    worker.finish_work()


def get_worker_class(module_name: Optional[str]) -> type(Worker):
    if module_name is None:
        raise Exception("Module name not found")

    module = importlib.import_module(module_name)

    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Worker) and obj is not Worker:
            return obj


def load_config(path: str) -> Box:
    training_config = {}
    with open(path, "r") as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)

    return Box(training_config)
