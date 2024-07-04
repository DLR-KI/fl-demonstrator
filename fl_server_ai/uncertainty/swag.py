# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Any, Dict, Tuple

from fl_server_core.models import SWAGModel

from .base import UncertaintyBase


class SWAG(UncertaintyBase):
    """
    Stochastic Weight Averaging Gaussian (SWAG) uncertainty estimation.
    """

    @classmethod
    def prediction(cls, input: torch.Tensor, model: SWAGModel) -> Tuple[torch.Tensor, Dict[str, Any]]:
        options = cls.get_options(model)
        N = options.get("N", 10)

        net: torch.nn.Module = model.get_torch_model()

        # first and second moment are already ensured to be in
        # alphabetical order in the database
        fm = model.first_moment
        sm = model.second_moment
        std = sm - torch.pow(fm, 2)
        params = torch.normal(mean=fm[None, :], std=std).expand(N, -1)

        prediction_list = []
        for n in range(N):
            torch.nn.utils.vector_to_parameters(params[n], net.parameters())
            prediction = net(input)
            prediction_list.append(prediction)
        predictions = torch.stack(prediction_list)

        inference = predictions.mean(dim=0)
        uncertainty = cls.interpret(predictions)
        return inference, uncertainty
