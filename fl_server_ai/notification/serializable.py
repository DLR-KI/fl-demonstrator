# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID


@dataclass
class Serializable:
    """
    Serializable `dataclass` base class which provides a method to serialize the dataclass instance into a dictionary.
    """

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the dataclass instance into a dictionary.

        UUID fields are converted to strings.

        Returns:
            dict[str, Any]: The serialized dataclass instance.
        """
        return {key: str(value) if isinstance(value, UUID) else value for key, value in asdict(self).items()}
