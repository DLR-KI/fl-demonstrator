from dataclasses import dataclass
from uuid import UUID

from fl_server_core.models import Training as TrainingDB

from ..serializable import Serializable
from ..notification_type import NotificationType
from .training import TrainingNotification


class TrainingFinishedNotification(TrainingNotification["TrainingFinishedNotification.Body"]):
    type: NotificationType = NotificationType.TRAINING_FINISHED

    @dataclass
    class Body(Serializable):
        global_model_uuid: UUID

    @classmethod
    def from_training(cls, training: TrainingDB):
        receivers = list(training.participants.all())
        if not receivers.__contains__(training.actor):
            receivers.append(training.actor)
        return cls(
            receivers=receivers,
            body=cls.Body(
                global_model_uuid=training.model.id
            ),
            training_uuid=training.id
        )
