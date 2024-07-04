# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.utils.translation import gettext_lazy as _
from enum import Enum
from typing import Optional


class NotificationType(Enum):
    """
    Notification types including a short description for each type.
    """

    UPDATE_ROUND_START = "UPDATE_ROUND_START", _("Start update round")
    SWAG_ROUND_START = "SWAG_ROUND_START", _("Start swag round")
    TRAINING_START = "TRAINING_START", _("Training start")
    TRAINING_FINISHED = "TRAINING_FINISHED", _("Training finished")
    CLIENT_REMOVED = "CLIENT_REMOVED", _("Client removed")
    MODEL_TEST_ROUND = "MODEL_TEST_ROUND", _("Start model round")

    def __new__(cls, *args, **kwargs):
        """
        Override the `__new__` method to set the value of the enum member.

        Args:
            *args: Variable length argument list where the first value is the value of the enum member.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: New instance of the class.
        """
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, description: Optional[str] = None):
        """
        Initialize the enum member.

        Args:
            _ (str): The value of the enum member. Ignored in this method as it's already set by __new__.
            description (Optional[str], optional): The description of the enum member. Defaults to None.
        """
        # ignore the first param since it's already set by __new__
        self._description_ = description

    @property
    def description(self):
        """
        Property to get the description of the enum member.

        Returns:
            str: The description of the enum member.
        """
        return self._description_
