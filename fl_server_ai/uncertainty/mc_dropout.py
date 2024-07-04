# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd
from typing import Any, Dict, Tuple

from fl_server_core.models import Model
from fl_server_core.utils.torch_serialization import is_torchscript_instance

from .base import UncertaintyBase


def set_dropout(model: Module, state: bool = True):
    """
    Set the state of the dropout layers to enable or disable them even during inference.

    Args:
        model (Module): PyTorch module
        state (bool, optional): Enable or disable dropout layers. Defaults to True.
    """
    is_torchscript_model = is_torchscript_instance(model)
    for m in model.modules():
        name = m.original_name if is_torchscript_model else m.__class__.__name__
        if isinstance(m, _DropoutNd) or name.lower().__contains__("dropout"):
            m.train(mode=state)


class MCDropout(UncertaintyBase):
    """
    Monte Carlo (MC) Dropout Uncertainty Estimation

    Requirements:

    - model with dropout layers
    - T, number of samples per input (number of monte-carlo samples/forward passes)

    References:

    - Paper: Understanding Measures of Uncertainty for Adversarial Example Detection
             <https://arxiv.org/abs/1803.08533>
    - Code inspiration: <https://github.com/lsgos/uncertainty-adversarial-paper/tree/master>
    """

    @classmethod
    def prediction(cls, input: Tensor, model: Model) -> Tuple[torch.Tensor, Dict[str, Any]]:
        options = cls.get_options(model)
        N = options.get("N", 10)
        softmax = options.get("softmax", False)

        net: Module = model.get_torch_model()
        net.eval()
        set_dropout(net, state=True)

        out_list = []
        for _ in range(N):
            output = net(input).detach()
            # convert to probabilities if necessary
            if softmax:
                output = torch.softmax(output, dim=1)
            out_list.append(output)
        out = torch.stack(out_list, dim=0)  # (n_mc, batch_size, n_classes)

        inference = out.mean(dim=0)
        uncertainty = cls.interpret(out)
        return inference, uncertainty
