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
    if training.uncertainty_method == UncertaintyMethod.SWAG:
        return SWAGModelTrainer
    if training.options.get("daisy_chain_period", 0) > 0:
        return FedDCModelTrainer
    return ModelTrainer


def get_trainer(training: Training, options: Optional[TrainerOptions] = None) -> "ModelTrainer":
    return get_trainer_class(training)(training, options)


class ModelTrainer:

    def __new__(cls, training: Training, options: Optional[TrainerOptions] = None) -> "ModelTrainer":
        return super().__new__(cls.get_trainer_class(training))

    @classmethod
    def get_trainer_class(cls, training: Training) -> Type["ModelTrainer"]:
        return get_trainer_class(training)

    def __init__(self, training: Training, options: Optional[TrainerOptions] = None):
        super().__init__()
        self.training = training
        self.options = options if options else TrainerOptions()

    def start(self):
        self.training.state = TrainingState.ONGOING
        self.training.save()
        TrainingStartNotification.from_training(self.training).send()
        TrainingRoundStartNotification.from_training(self.training).send()

    def finish(self):
        self.training.state = TrainingState.COMPLETED
        self.training.save()
        TrainingFinishedNotification.from_training(self.training).send()

    def start_round(self):
        TrainingRoundStartNotification.from_training(self.training).send()

    def test_round(self):
        TrainingModelTestNotification.from_training(self.training).send()

    def handle(self, event: ModelTrainerEvent):
        event.handle()
        event.next()

    def handle_cls(self, event_cls: Type[ModelTrainerEvent]):
        self.handle(event_cls(self))


class SWAGModelTrainer(ModelTrainer):

    def start_swag_round(self):
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
