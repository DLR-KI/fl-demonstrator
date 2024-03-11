from abc import ABC, abstractmethod
from logging import getLogger

from .. import model_trainer


class ModelTrainerEvent(ABC):

    _logger = getLogger("fl.server")

    def __init__(self, trainer: "model_trainer.ModelTrainer"):
        super().__init__()
        self.trainer = trainer
        self.training = trainer.training

    @abstractmethod
    def handle(self):
        pass

    @abstractmethod
    def next(self):
        pass
