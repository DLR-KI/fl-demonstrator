# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from .base import Aggregation
from .mean import MeanAggregation
from .method import get_aggregation_class


__all__ = ["Aggregation", "get_aggregation_class", "MeanAggregation"]
