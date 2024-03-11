import torch
from typing import Any, Dict, Tuple

from fl_server_core.models import MeanModel

from .base import UncertaintyBase


class Ensemble(UncertaintyBase):

    @classmethod
    def prediction(cls, input: torch.Tensor, model: MeanModel) -> Tuple[torch.Tensor, Dict[str, Any]]:
        output_list = []
        for m in model.models.all():
            net = m.get_torch_model()
            output = net(input).detach()
            output_list.append(output)
        outputs = torch.stack(output_list, dim=0)  # (N, batch_size, n_classes)  # N = number of models

        inference = outputs.mean(dim=0)
        uncertainty = cls.interpret(outputs)
        return inference, uncertainty
