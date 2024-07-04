# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from celery import Signature
from celery.utils.log import get_task_logger
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from fl_server_core.models import Training as TrainingDB, User as UserDB
from fl_server_core.models.training import TrainingState
from fl_server_core.models.user import NotificationReceiver

from ...celery_tasks import app
from ...exceptions import ClientNotificationRejectionException, NotificationException
from ..serializable import Serializable
from ..notification_type import NotificationType
from .training import TrainingNotification


@app.task(bind=False, ignore_result=True)
def training_notification_callback_success(receiver: NotificationReceiver, training_uuid: UUID):
    """
    Callback that logs a successful training notification.

    Args:
        receiver (NotificationReceiver): The receiver of the notification.
        training_uuid (UUID): The UUID of the training.
    """
    logger = get_task_logger("fl.celery")
    logger.debug(f"Training {training_uuid}: User {receiver.id} accepted notification")


@app.task(bind=False, ignore_result=True)
def training_notification_callback_failure(exception: NotificationException, training_uuid: UUID):
    """
    Callback that logs a failed training notification.

    Also sets the training to an error state if the exception is not a client rejection.

    Args:
        exception (NotificationException): The exception that occurred.
        training_uuid (UUID): The UUID of the training.
    """
    logger = get_task_logger("fl.celery")
    if isinstance(exception, ClientNotificationRejectionException):
        # client sent error response, remove
        receiver: NotificationReceiver = exception.notification_return_object
        logger.warn(
            f"Training {training_uuid}: User {receiver.id} sent error response: {exception.status_code}."
            "User will be removed from training!"
        )
        user = receiver if isinstance(receiver, UserDB) else UserDB.objects.get(id=receiver.id)
        TrainingDB.objects.get(id=training_uuid).participants.remove(user)
    else:
        # set training to error state
        e = exception.inner_exception if exception.inner_exception else exception
        logger.error(f"Training {training_uuid}: Exception occurred during sending: {e}")
        logger.error(f"Training {training_uuid} is in error state!")
        training = TrainingDB.objects.get(id=training_uuid)
        training.state = TrainingState.ERROR
        training.save()


class TrainingRoundStartNotification(TrainingNotification["TrainingRoundStartNotification.Body"]):
    """
    Notification for the start of a training round.
    """

    type: NotificationType = NotificationType.UPDATE_ROUND_START
    """The type of the notification."""

    @property
    def callback_success(self) -> Optional[Signature]:
        return training_notification_callback_success.s(training_uuid=self.training_uuid)

    @property
    def callback_error(self) -> Optional[Signature]:
        return training_notification_callback_failure.s(training_uuid=self.training_uuid)

    @dataclass
    class Body(Serializable):
        """
        Inner class representing the body of the notification.
        """
        round: int
        """The round number."""
        global_model_uuid: UUID
        """The UUID of the global model."""

    @classmethod
    def from_training(cls, training: TrainingDB):
        """
        Create a `TrainingRoundStartNotification` instance from a training object.

        Args:
            training (TrainingDB): The training object to create the notification from.

        Returns:
            TrainingRoundStartNotification: The created notification.
        """
        return cls(
            receivers=training.participants.all(),
            body=cls.Body(
                round=training.model.round,
                global_model_uuid=training.model.id
            ),
            training_uuid=training.id
        )
