# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Generic
from uuid import UUID

from ..notification import Notification, TBody


@dataclass
class TrainingNotification(Generic[TBody], Notification[TBody]):
    """
    Abstract base class for training notifications.
    """

    training_uuid: UUID
    """The UUID of the training."""

    def serialize(self) -> dict[str, Any]:
        data = super().serialize()
        data["training_uuid"] = str(self.training_uuid)
        return data
