# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from uuid import UUID

from fl_server_core.models import Training as TrainingDB

from ..serializable import Serializable
from ..notification_type import NotificationType
from .training import TrainingNotification


class TrainingStartNotification(TrainingNotification["TrainingStartNotification.Body"]):
    """
    Class representing a notification for the start of a training.
    """

    type: NotificationType = NotificationType.TRAINING_START
    """The type of the notification."""

    @dataclass
    class Body(Serializable):
        """
        Inner class representing the body of the notification.
        """
        global_model_uuid: UUID
        """The UUID of the global model."""

    @classmethod
    def from_training(cls, training: TrainingDB):
        """
        Create a `TrainingStartNotification` instance from a training object.

        Args:
            training (TrainingDB): The training object to create the notification from.

        Returns:
            TrainingStartNotification: The created notification.
        """
        return cls(
            receivers=training.participants.all(),
            body=cls.Body(
                global_model_uuid=training.model.id
            ),
            training_uuid=training.id
        )
