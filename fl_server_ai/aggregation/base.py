# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from logging import getLogger
import torch
from typing import Sequence


class Aggregation(ABC):
    """
    Abstract base class for aggregation strategies.
    """
    _logger = getLogger("fl.server")

    @abstractmethod
    def aggregate(
        self,
        models: Sequence[torch.nn.Module],
        model_sample_sizes: Sequence[int],
        *,
        deepcopy: bool = True
    ) -> torch.nn.Module:
        """
        Abstract method for aggregating models.

        Args:
            models (Sequence[torch.nn.Module]): The models to be aggregated.
            model_sample_sizes (Sequence[int]): The sample sizes for each model.
            deepcopy (bool, optional): Whether to create a deep copy of the models. Defaults to True.

        Returns:
            torch.nn.Module: The aggregated model.
        """
        pass
