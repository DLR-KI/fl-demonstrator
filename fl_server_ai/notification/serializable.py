from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID


@dataclass
class Serializable:

    def serialize(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, UUID) else value for key, value in asdict(self).items()}
