import torch
from typing import Sequence

from .base import Aggregation


class FedDC(Aggregation):
    """
    FedDC (Federated daisy-chaining)

    To tackle the problem that client models are potentially quite small and thus the models tend to overfit and
    therefore result in bad prediction quality on unseen data, one proposed solution is
    FedDC (also named Federated daisy-chaining).
    FedDC sends before each aggregation step each client model to another randomly selected client, which trains
    it on its local data.
    From the model perspective, it is as if the model is trained on a larger dataset.

    Paper: [Picking Daisies in Private: Federated Learning from Small Datasets](https://openreview.net/forum?id=GVDwiINkMR)
    """  # noqa: E501

    @torch.no_grad()
    def aggregate(
        self,
        models: Sequence[torch.nn.Module],
        model_sample_sizes: Sequence[int],
        *,
        deepcopy: bool = True
    ) -> torch.nn.Module:
        raise NotImplementedError("FedDC is not implemented yet!")
