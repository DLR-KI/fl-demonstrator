# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from .mean import MeanAggregation


class FedProx(MeanAggregation):
    """
    To tackle the problem that client models drift away from optimum due to data heterogeneity,
    different learning speeds, etc. one proposed solution is FedProx.
    FedProx limits the client drift by using a modified learning objective but keeping the standard
    FedAvg aggregation method.

    Note:
    FedProx does not do anything different on the server side than normal FedAvg.
    The difference lies in the application of a special loss function on the client side.

    Paper: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
    """
    pass
