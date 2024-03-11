from django.utils.translation import gettext_lazy as _
from enum import Enum
from typing import Optional


class NotificationType(Enum):
    UPDATE_ROUND_START = "UPDATE_ROUND_START", _("Start update round")
    SWAG_ROUND_START = "SWAG_ROUND_START", _("Start swag round")
    TRAINING_START = "TRAINING_START", _("Training start")
    TRAINING_FINISHED = "TRAINING_FINISHED", _("Training finished")
    CLIENT_REMOVED = "CLIENT_REMOVED", _("Client removed")
    MODEL_TEST_ROUND = "MODEL_TEST_ROUND", _("Start model round")

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, description: Optional[str] = None):
        # ignore the first param since it's already set by __new__
        self._description_ = description

    @property
    def description(self):
        return self._description_
