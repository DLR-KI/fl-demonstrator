from dataclasses import dataclass
from typing import Any, Generic
from uuid import UUID

from ..notification import Notification, TBody


@dataclass
class TrainingNotification(Generic[TBody], Notification[TBody]):
    training_uuid: UUID

    def serialize(self) -> dict[str, Any]:
        data = super().serialize()
        data["training_uuid"] = str(self.training_uuid)
        return data
