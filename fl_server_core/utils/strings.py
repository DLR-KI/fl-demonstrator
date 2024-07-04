# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional


def str2bool(value: Optional[str], fallback: Optional[bool] = None) -> bool:
    """
    Convert string to boolean.

    Sources: <https://stackoverflow.com/a/43357954>

    Args:
        value (str): value to convert
        fallback (Optional[bool], optional): Fallback if value can not be converted.
            If None raise ValueError. Defaults to None.

    Raises:
        ValueError: value can not be converted and no fallback is specified

    Returns:
        bool: boolean interpretation of value
    """
    if value is not None:
        if isinstance(value, bool):
            return value
        if value.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif value.lower() in ("no", "false", "f", "n", "0"):
            return False
    if fallback is None:
        raise ValueError(f"Can not convert '{value}' to boolean.")
    return fallback
