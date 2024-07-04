# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta
from celery import Signature
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from dataclasses import dataclass, field
from rest_framework import status
import requests
import time
from typing import Any, Generic, List, Optional, TypeVar

from fl_server_core.models.user import NotificationReceiver

from ..celery_tasks import app
from ..exceptions import ClientNotificationRejectionException, NotificationException

from .serializable import Serializable
from .notification_type import NotificationType


TBody = TypeVar("TBody", bound=Serializable)
TReturn = TypeVar("TReturn", bound=Any)


@app.task(bind=False, ignore_result=False)
def send_notifications(
    notification: "Notification",
    callback_success: Optional[Signature] = None,
    callback_error: Optional[Signature] = None
):
    """
    Send a single notification to all receivers.

    Args:
        notification (Notification): The notification to be sent.
        callback_success (Optional[Signature], optional): The callback to be called on success. Defaults to None.
        callback_error (Optional[Signature], optional): The callback to be called on error. Defaults to None.
    """
    logger = get_task_logger("fl.celery")
    for receiver in notification.receivers:
        logger.info(
            f"Sending notification '{notification.type}' " +
            f"to client '{receiver.id}' with URL '{receiver.message_endpoint}'"
        )
        send_notification.s(
            endpoint_url=receiver.message_endpoint, json_data=notification.serialize(), return_obj=receiver
        ).apply_async(link=callback_success, link_error=callback_error, retry=False)


@app.task(bind=False, ignore_result=False)
def send_notification(
    endpoint_url: str,
    json_data: Any,
    max_retries: int = 5,
    *,
    return_obj: Optional[TReturn] = None
) -> Optional[TReturn]:
    """
    Send a single notification to a specified endpoint URL.

    Args:
        endpoint_url (str): The endpoint URL to send the notification to.
        json_data (Any): The JSON data to be sent.
        max_retries (int, optional): The maximum number of retries. Defaults to 5.
        return_obj (Optional[TReturn], optional): The object to return. Defaults to None.

    Returns:
        Optional[TReturn]: The object specified by return_obj, or None if return_obj is None.

    Raises:
        ClientNotificationRejectionException: If the server returns a non-success status code.
        NotificationException: If an exception occurs while sending the notification.
    """
    logger = get_task_logger("fl.celery")
    retries: int = 0
    while True:
        try:
            response = requests.post(url=endpoint_url, json=json_data, headers={"content-type": "application/json"})
            if not status.is_success(response.status_code):
                raise ClientNotificationRejectionException(response, endpoint_url, json_data, max_retries, return_obj)
            return return_obj
        except ClientNotificationRejectionException as e:
            logger.error(f"Sending notification to URL '{endpoint_url}' return status code: {e.status_code}")
            raise e
        except Exception as e:
            logger.warn(f"Sending notification to URL '{endpoint_url}' produced exception: {e}")
            retries += 1
            if retries > max_retries:
                raise NotificationException(endpoint_url, json_data, max_retries, return_obj, e)
            time.sleep(5*retries)


@dataclass
class Notification(Generic[TBody], Serializable, metaclass=ABCMeta):
    """
    Abstract base class for notifications.
    """

    receivers: List[NotificationReceiver]
    """The receivers of the notification."""
    body: TBody
    """The body of the notification."""
    type: NotificationType = field(init=False)
    """The type of the notification."""

    @dataclass
    class Body(Serializable):
        """
        Inner class for the body of the notification.
        """
        pass

    @property
    def callback_success(self) -> Optional[Signature]:
        """
        The callback to be called on success. By default, this is None.

        Returns:
            Optional[Signature]: The callback to be called on success, or None if no such callback is set.
        """
        return None

    @property
    def callback_error(self) -> Optional[Signature]:
        """
        The callback to be called on error. By default, this is None.

        Returns:
            Optional[Signature]: The callback to be called on error, or None if no such callback is set.
        """
        return None

    def send(
        self,
        callback_success: Optional[Signature] = None,
        callback_error: Optional[Signature] = None
    ) -> AsyncResult:
        """
        Send notification to the receivers asynchronously.

        Args:
            callback_success (Optional[Signature], optional): The callback to be called on success. Defaults to None.
            callback_error (Optional[Signature], optional): The callback to be called on error. Defaults to None.

        Returns:
            AsyncResult: The result of the asynchronous operation.
        """
        callback_success = callback_success or self.callback_success
        callback_error = callback_error or self.callback_error
        return send_notifications.s(
            notification=self, callback_success=callback_success, callback_error=callback_error
        ).apply_async(retry=False)

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the notification into a dictionary.

        Returns:
            dict[str, Any]: The serialized notification.
        """
        return {
            "notification_type": self.type.value,
            "body": self.body.serialize()
        }
