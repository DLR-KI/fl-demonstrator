from dataclasses import dataclass
from uuid import UUID

from fl_server_core.models import Training as TrainingDB

from ..serializable import Serializable
from ..notification_type import NotificationType
from .training import TrainingNotification


class TrainingStartNotification(TrainingNotification["TrainingStartNotification.Body"]):
    type: NotificationType = NotificationType.TRAINING_START

    @dataclass
    class Body(Serializable):
        global_model_uuid: UUID

    @classmethod
    def from_training(cls, training: TrainingDB):
        return cls(
            receivers=training.participants.all(),
            body=cls.Body(
                global_model_uuid=training.model.id
            ),
            training_uuid=training.id
        )
