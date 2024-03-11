import torch
from typing import Any, Dict, Tuple

from fl_server_core.models import Model

from .base import UncertaintyBase


class NoneUncertainty(UncertaintyBase):

    @classmethod
    def prediction(cls, input: torch.Tensor, model: Model) -> Tuple[torch.Tensor, Dict[str, Any]]:
        net: torch.nn.Module = model.get_torch_model()
        prediction: torch.Tensor = net(input)
        return prediction.argmax(dim=1), {}
