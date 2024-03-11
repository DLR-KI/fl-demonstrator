from django.test import TestCase
import pickle
import torch
from unittest.mock import patch

from fl_server_core.models.model import SWAGModel
from fl_server_core.models.training import TrainingState, UncertaintyMethod
from fl_server_core.tests.dummy import Dummy

from ..notification.training import TrainingSWAGRoundStartNotification
from ..trainer.events import SWAGRoundFinished, TrainingRoundFinished
from ..trainer.model_trainer import get_trainer, get_trainer_class, SWAGModelTrainer
from ..uncertainty import SWAG


class SwagTest(TestCase):

    def test_trainer_class(self):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.SWAG)
        trainer_cls = get_trainer_class(training)
        self.assertTrue(trainer_cls is SWAGModelTrainer)

    def test_trainer_type(self):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.SWAG)
        trainer = get_trainer(training)
        self.assertTrue(type(trainer) is SWAGModelTrainer)

    @patch.object(TrainingSWAGRoundStartNotification, "send")
    def test_start_swag_round(self, send_method):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.SWAG)
        trainer = get_trainer(training)
        assert type(trainer) is SWAGModelTrainer
        trainer.start_swag_round()
        self.assertEqual(TrainingState.SWAG_ROUND, training.state)
        self.assertTrue(send_method.called)

    @patch.object(TrainingRoundFinished, "handle")
    @patch.object(TrainingRoundFinished, "next")
    @patch.object(TrainingSWAGRoundStartNotification, "send")
    def test_start_swag_round_via_handle(self, send_method, next_method, handle_method):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.SWAG)
        trainer = get_trainer(training)
        assert type(trainer) is SWAGModelTrainer
        event = TrainingRoundFinished(trainer)
        trainer.handle(event)
        self.assertEqual(TrainingState.SWAG_ROUND, training.state)
        self.assertTrue(handle_method.called)
        self.assertFalse(next_method.called)
        self.assertTrue(send_method.called)

    @patch.object(SWAGRoundFinished, "handle")
    @patch.object(TrainingRoundFinished, "next")
    def test_handle_swag_round_finished(self, base_cls_next_method, handle_method):
        training = Dummy.create_training(uncertainty_method=UncertaintyMethod.SWAG)
        trainer = get_trainer(training)
        assert type(trainer) is SWAGModelTrainer
        event = SWAGRoundFinished(trainer)
        trainer.handle(event)
        self.assertEqual(TrainingState.ONGOING, training.state)
        self.assertTrue(handle_method.called)
        self.assertTrue(base_cls_next_method.called)

    def test_prediction(self):
        model = pickle.dumps(torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Tanh()))
        first_moment = pickle.dumps(torch.tensor([1.0, 0.0]))
        second_moment = pickle.dumps(torch.tensor([1.0, 0.0]))
        model = Dummy.create_model(SWAGModel, weights=model, swag_first_moment=first_moment,
                                   swag_second_moment=second_moment)
        Dummy.create_training(
            model=model,
            uncertainty_method=UncertaintyMethod.SWAG,
            options=dict(uncertainty={"N": 10})
        )
        X = torch.tensor([[-4.0], [-2.0], [2.0], [4.0]])
        y = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        logits, _ = SWAG.prediction(X, model)
        torch.testing.assert_close(y, torch.sign(torch.squeeze(logits)))
