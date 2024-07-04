# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from .mean import MeanAggregation


class FedDC(MeanAggregation):
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
    pass
