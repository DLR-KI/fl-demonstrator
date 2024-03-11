from django.test import TestCase
import pickle
import torch

from fl_server_core.models.model import GlobalModel
from fl_server_core.models.training import UncertaintyMethod
from fl_server_core.tests.dummy import Dummy

from ..trainer.model_trainer import get_trainer, get_trainer_class, ModelTrainer
from ..uncertainty import MCDropout


class MCDropoutTest(TestCase):

    def test_trainer_class(self):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.MC_DROPOUT)
        trainer_cls = get_trainer_class(training)
        self.assertTrue(trainer_cls is ModelTrainer)

    def test_trainer_type(self):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.MC_DROPOUT)
        trainer = get_trainer(training)
        self.assertTrue(type(trainer) is ModelTrainer)

    def test_prediction(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(1, 1)
        layer.weight = torch.nn.Parameter(torch.tensor([[1.]]))
        layer.bias = torch.nn.Parameter(torch.tensor([0.]))
        model = pickle.dumps(torch.nn.Sequential(layer, torch.nn.Dropout(0.5), torch.nn.Tanh()))
        model = Dummy.create_model(GlobalModel, weights=model)
        Dummy.create_training(
            model=model,
            uncertainty_method=UncertaintyMethod.MC_DROPOUT
        )
        X = torch.tensor([[-4.0], [-2.0], [2.0], [4.0]])
        y = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        logits, uncertainty_dict = MCDropout.prediction(X, model)
        torch.testing.assert_close(y, torch.sign(torch.squeeze(logits)))
        torch.testing.assert_close(torch.tensor([[0.2667], [0.2330], [0.2663], [0.2333]]),
                                   uncertainty_dict["variance"], atol=1e-4, rtol=0.001)
        torch.testing.assert_close(torch.tensor([[0.5164], [0.4827], [0.5161], [0.4830]]),
                                   uncertainty_dict["std"], atol=1e-4, rtol=0.001)
        self.assertFalse("predictive_entropy" in uncertainty_dict)
        self.assertFalse("expected_entropy" in uncertainty_dict)
        self.assertFalse("mutual_info" in uncertainty_dict)
