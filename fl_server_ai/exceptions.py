# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from requests import Response
from typing import Any, Optional


class AggregationException(Exception):
    """
    Exception raised for errors in the aggregation process.

    This is a custom exception class that should be raised when there is an error during the aggregation process.
    """
    pass


class NotificationException(Exception):
    """
    Exception raised for errors in the notification process.

    This is a custom exception class that should be raised when there is an error during the notification process.
    """

    def __init__(
        self,
        endpoint_url: str,
        json_data: Any,
        max_retries: int,
        return_obj: Any = None,
        inner_exception: Optional[Exception] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.endpoint_url = endpoint_url
        """The URL of the endpoint that the notification was sent to."""
        self.json_data = json_data
        """The JSON data that was sent in the notification."""
        self.max_retries = max_retries
        """The maximum number of times to retry the notification."""
        self.notification_return_object = return_obj
        """The object to return if the notification fails."""
        self.inner_exception = inner_exception
        """The original exception that caused the notification error."""


class ClientNotificationRejectionException(NotificationException):
    """
    Exception raised when a client rejects a notification.

    This is a custom exception class that should be raised when a client rejects a notification.
    It inherits from NotificationException and adds additional attributes related to the client's response.
    """

    def __init__(self, response: Response, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response
        """The client's response to the notification."""
        self.status_code = response.status_code
        """The HTTP status code of the client's response."""
