# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Dict


def parse(d: Dict) -> SimpleNamespace:
    """
    Parse nested dict to namespace to support dot notation/access.

    Args:
        d (Dict): dictionary to parse

    Returns:
        SimpleNamespace: dict as namespace
    """
    x = SimpleNamespace()
    [setattr(  # type:ignore[func-returns-value]
        x, k,
        parse(v) if isinstance(v, dict) else [parse(e) for e in v] if isinstance(v, list) else v
    ) for k, v in d.items()]
    return x
