from django.test import TestCase
import torch

from ..aggregation.mean import MeanAggregation


def _create_model_and_init(init: float) -> torch.nn.Module:
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
    return model


class AggregationTest(TestCase):

    def test_aggregate(self):
        aggr = MeanAggregation()
        models = [_create_model_and_init(i) for i in range(10)]
        res = aggr.aggregate(models, [1]*10)
        self.assertTrue(isinstance(res, torch.nn.Sequential))
        self.assertEqual(len(list(models[0].parameters())), len(list(res.parameters())))
        torch.testing.assert_close(res[0].weight, torch.ones_like(res[0].weight) * 4.5)
        torch.testing.assert_close(res[3].weight, torch.ones_like(res[3].weight) * 4.5)

    def test_aggregate_sample_sizes(self):
        aggr = MeanAggregation()
        models = [_create_model_and_init(i) for i in range(3)]
        res = aggr.aggregate(models, [0, 1, 2])
        self.assertTrue(isinstance(res, torch.nn.Sequential))
        self.assertEqual(len(list(models[0].parameters())), len(list(res.parameters())))
        torch.testing.assert_close(res[0].weight, torch.ones_like(res[0].weight) * (5/3))
        torch.testing.assert_close(res[3].weight, torch.ones_like(res[3].weight) * (5/3))
