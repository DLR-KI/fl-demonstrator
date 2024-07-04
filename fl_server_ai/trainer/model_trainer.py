# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from random import shuffle
from typing import Optional, Type

from fl_server_core.models.model import LocalModel
from fl_server_core.models.training import Training, TrainingState, UncertaintyMethod
from fl_server_ai.notification.training import (
    TrainingFinishedNotification,
    TrainingModelTestNotification,
    TrainingRoundStartNotification,
    TrainingStartNotification,
    TrainingSWAGRoundStartNotification,
)

from .events import ModelTrainerEvent, DaisyChainRoundFinished, TrainingRoundFinished
from .options import TrainerOptions


def get_trainer_class(training: Training) -> Type["ModelTrainer"]:
    """
    Get the appropriate model trainer class based on a training object.

    Args:
        training (Training): The training for which to get the trainer class.

    Returns:
        Type["ModelTrainer"]: The appropriate trainer class.
    """
    if training.uncertainty_method == UncertaintyMethod.SWAG:
        return SWAGModelTrainer
    if training.options.get("daisy_chain_period", 0) > 0:
        return FedDCModelTrainer
    return ModelTrainer


def get_trainer(training: Training, options: Optional[TrainerOptions] = None) -> "ModelTrainer":
    """
    Get a model trainer instance for the given training object.

    Args:
        training (Training): The training for which to get the trainer.
        options (Optional[TrainerOptions]): The options for the trainer. Defaults to None.

    Returns:
        "ModelTrainer": The trainer instance.
    """
    return get_trainer_class(training)(training, options)


class ModelTrainer:
    """
    Common model trainer.
    """

    def __new__(cls, training: Training, options: Optional[TrainerOptions] = None) -> "ModelTrainer":
        """
        Ensure that the correct trainer class is returned based on the training.

        The returned trainer class is determined by the `get_trainer_class` method.
        It could for example return a `SWAGModelTrainer` if the training uses the SWAG uncertainty method.

        Args:
            training (Training): The training for which to get the trainer.
            options (Optional[TrainerOptions]): The options for the trainer. If None, default options will be used.

        Returns:
            "ModelTrainer": The trainer instance.
        """
        return super().__new__(cls.get_trainer_class(training))

    @classmethod
    def get_trainer_class(cls, training: Training) -> Type["ModelTrainer"]:
        """
        Get the appropriate model trainer class based on a training object.

        Args:
            training (Training): The training for which to get the trainer class.

        Returns:
            Type["ModelTrainer"]: The appropriate trainer class.
        """
        return get_trainer_class(training)

    def __init__(self, training: Training, options: Optional[TrainerOptions] = None):
        """
        Initialize the trainer with the given training and options.

        Args:
            training (Training): The training to be handled by the trainer.
            options (Optional[TrainerOptions]): The options for the trainer. If None, default options will be used.
        """
        super().__init__()
        self.training = training
        """The training to be handled by the trainer."""
        self.options = options if options else TrainerOptions()
        """The options for the trainer."""

    def start(self):
        """
        Start the training and send a start notification.
        """
        self.training.state = TrainingState.ONGOING
        self.training.save()
        TrainingStartNotification.from_training(self.training).send()
        TrainingRoundStartNotification.from_training(self.training).send()

    def finish(self):
        """
        Finish the training and send a finished notification.
        """
        self.training.state = TrainingState.COMPLETED
        self.training.save()
        TrainingFinishedNotification.from_training(self.training).send()

    def start_round(self):
        """
        Start a training round and send a round start notification.
        """
        TrainingRoundStartNotification.from_training(self.training).send()

    def test_round(self):
        """
        Test a training round and send a model test notification.
        """
        TrainingModelTestNotification.from_training(self.training).send()

    def handle(self, event: ModelTrainerEvent):
        """
        Handle a model trainer event and proceed to the next event.

        Args:
            event (ModelTrainerEvent): The event to handle.
        """
        event.handle()
        event.next()

    def handle_cls(self, event_cls: Type[ModelTrainerEvent]):
        """
        Handle a model trainer event class by creating an instance of the event and handling it.

        Args:
            event_cls (Type[ModelTrainerEvent]): The class of the event to handle.
        """
        self.handle(event_cls(self))


class SWAGModelTrainer(ModelTrainer):
    """
    Stochastic weight averaging Gaussian (SWAG) model trainer.
    """

    def start_swag_round(self):
        """
        Start a SWAG round and send a SWAG round start notification.
        """
        self.training.state = TrainingState.SWAG_ROUND
        self.training.save()
        TrainingSWAGRoundStartNotification.from_training(self.training).send()

    def handle(self, event: ModelTrainerEvent):
        event.handle()
        if type(event) is TrainingRoundFinished:
            self.start_swag_round()
        else:
            event.next()


class FedDCModelTrainer(ModelTrainer):
    """
    Federated daisy-chaining (FedDC) model trainer.

    To tackle the problem that client models are potentially quite small and thus the models tend to overfit and
    therefore result in bad prediction quality on unseen data, one proposed solution is
    FedDC (also named Federated daisy-chaining).
    FedDC sends before each aggregation step each client model to another randomly selected client, which trains
    it on its local data.
    From the model perspective, it is as if the model is trained on a larger dataset.

    Paper: [Picking Daisies in Private: Federated Learning from Small Datasets](https://openreview.net/forum?id=GVDwiINkMR)
    """  # noqa: E501

    def start_round(self):
        dc_period = self.training.options.get("daisy_chain_period", 0)
        if dc_period < 1 or self.training.model.round % dc_period == 0:
            # start training round, first (local) training round, therefore no local models to permute
            TrainingRoundStartNotification.from_training(self.training).send()
            return

        # daily chaining period, therefore send the permutated client models back for further training
        clients = self.training.participants.all()
        model_ids = list(LocalModel.objects.filter(
            base_model=self.training.model,
            round=self.training.model.round - 1
        ).values_list("pk", flat=True))
        shuffle(model_ids)
        for client, model_id in zip(clients, model_ids):
            TrainingRoundStartNotification(
                receivers=[client],
                body=TrainingRoundStartNotification.Body(
                    round=self.training.model.round,
                    global_model_uuid=model_id
                ),
                training_uuid=self.training.id
            ).send()

    def handle(self, event: ModelTrainerEvent):
        if type(event) is TrainingRoundFinished:
            real_event = DaisyChainRoundFinished(self)
            real_event.handle()
            real_event.next()
        else:
            event.handle()
            event.next()
