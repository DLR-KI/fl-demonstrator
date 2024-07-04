# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from logging import getLogger

from .. import model_trainer


class ModelTrainerEvent(ABC):
    """
    Abstract base class for a model trainer event.
    """

    _logger = getLogger("fl.server")

    def __init__(self, trainer: "model_trainer.ModelTrainer"):
        """
        Initialize the event with the given trainer.

        Args:
            trainer (model_trainer.ModelTrainer): The trainer that the event is associated with.
        """
        super().__init__()
        self.trainer = trainer
        """The trainer that the event is associated with."""
        self.training = trainer.training
        """The training that the event is associated with."""

    @abstractmethod
    def handle(self):
        """
        Handle the event.
        """
        pass

    @abstractmethod
    def next(self):
        """
        Proceed with the next event.
        """
        pass
