# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
import torch

from fl_server_core.models.model import GlobalModel, MeanModel
from fl_server_core.models.training import UncertaintyMethod
from fl_server_core.tests.dummy import Dummy
from fl_server_core.utils.torch_serialization import from_torch_module

from ..trainer.model_trainer import get_trainer, get_trainer_class, ModelTrainer
from ..uncertainty import Ensemble


class EnsembleTest(TestCase):

    def test_trainer_class(self):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.ENSEMBLE)
        trainer_cls = get_trainer_class(training)
        self.assertTrue(trainer_cls is ModelTrainer)

    def test_trainer_type(self):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.ENSEMBLE)
        trainer = get_trainer(training)
        self.assertTrue(type(trainer) is ModelTrainer)

    def test_prediction(self):
        layer = torch.nn.Linear(1, 1)
        layer.weight = torch.nn.Parameter(torch.tensor([[1.]]))
        layer.bias = torch.nn.Parameter(torch.tensor([0.]))
        models = [
            from_torch_module(torch.jit.script(torch.nn.Sequential(layer, torch.nn.Tanh())))
            for _ in range(10)
        ]
        models_db = [Dummy.create_model(GlobalModel, weights=model) for model in models]
        model = Dummy.create_model(MeanModel)
        model.models.set(models_db)
        Dummy.create_training(
            model=model,
            uncertainty_method=UncertaintyMethod.ENSEMBLE
        )
        X = torch.tensor([[-4.0], [-2.0], [2.0], [4.0]])
        y = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        logits, uncertainty_dict = Ensemble.prediction(X, model)
        torch.testing.assert_close(y, torch.sign(torch.squeeze(logits)))
        torch.testing.assert_close(torch.tensor([[0.]] * 4), uncertainty_dict["variance"])
        torch.testing.assert_close(torch.tensor([[0.]] * 4), uncertainty_dict["std"])
