# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Any, Dict, Tuple

from fl_server_core.models import Model

from .base import UncertaintyBase


class NoneUncertainty(UncertaintyBase):
    """
    Empty uncertainty estimation when no specific uncertainty method is used.

    This class does not calculate any uncertainty and only returns the prediction with an empty uncertainty dictionary.
    """

    @classmethod
    def prediction(cls, input: torch.Tensor, model: Model) -> Tuple[torch.Tensor, Dict[str, Any]]:
        net: torch.nn.Module = model.get_torch_model()
        prediction: torch.Tensor = net(input)
        return prediction.argmax(dim=1), {}
