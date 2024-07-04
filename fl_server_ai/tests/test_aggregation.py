# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
import torch

from fl_server_core.utils.torch_serialization import is_torchscript_instance

from ..aggregation.mean import MeanAggregation


def _create_torchscript_model_and_init(init: float) -> torch.jit.ScriptModule:
    init = float(init)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 5),
        torch.nn.Tanh(),
        torch.nn.BatchNorm1d(5),
        torch.nn.Linear(5, 3)
    )
    torch.nn.init.constant_(model[0].weight, init)
    torch.nn.init.constant_(model[0].bias, init)
    torch.nn.init.constant_(model[3].weight, init)
    torch.nn.init.constant_(model[3].bias, init)
    return torch.jit.script(model)


class AggregationTest(TestCase):

    def test_aggregate(self):
        aggr = MeanAggregation()
        models = [_create_torchscript_model_and_init(i) for i in range(10)]
        model = aggr.aggregate(models, [1]*10)
        cls_name = model.original_name if is_torchscript_instance(model) else model.__class__.__name__
        self.assertEqual(cls_name, "Sequential")
        res = model.state_dict()
        self.assertEqual(len(models[0].state_dict()), len(res))
        torch.testing.assert_close(res["0.weight"], torch.ones_like(res["0.weight"]) * 4.5)
        torch.testing.assert_close(res["3.weight"], torch.ones_like(res["3.weight"]) * 4.5)

    def test_aggregate_sample_sizes(self):
        aggr = MeanAggregation()
        models = [_create_torchscript_model_and_init(i) for i in range(3)]
        model = aggr.aggregate(models, [0, 1, 2])
        cls_name = model.original_name if is_torchscript_instance(model) else model.__class__.__name__
        self.assertEqual(cls_name, "Sequential")
        self.assertEqual(len(list(models[0].parameters())), len(list(model.parameters())))
        res = model.state_dict()
        torch.testing.assert_close(res["0.weight"], torch.ones_like(res["0.weight"]) * (5/3))
        torch.testing.assert_close(res["3.weight"], torch.ones_like(res["3.weight"]) * (5/3))
