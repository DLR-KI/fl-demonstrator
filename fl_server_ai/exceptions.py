from requests import Response
from typing import Any, Optional


class AggregationException(Exception):
    pass


class NotificationException(Exception):

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
        self.json_data = json_data
        self.max_retries = max_retries
        self.notification_return_object = return_obj
        self.inner_exception = inner_exception


class ClientNotificationRejectionException(NotificationException):

    def __init__(self, response: Response, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response
        self.status_code = response.status_code
