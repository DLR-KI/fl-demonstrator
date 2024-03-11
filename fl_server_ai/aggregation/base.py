from abc import ABC, abstractmethod
from logging import getLogger
import torch
from typing import Sequence


class Aggregation(ABC):
    _logger = getLogger("fl.server")

    @abstractmethod
    def aggregate(
        self,
        models: Sequence[torch.nn.Module],
        model_sample_sizes: Sequence[int],
        *,
        deepcopy: bool = True
    ) -> torch.nn.Module:
        pass
