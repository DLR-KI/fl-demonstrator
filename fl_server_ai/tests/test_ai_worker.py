# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
import torch
from unittest.mock import MagicMock, patch

from fl_server_core.models.model import SWAGModel
from fl_server_core.models.training import Training, TrainingState, UncertaintyMethod
from fl_server_core.tests import Dummy
from fl_server_core.utils.torch_serialization import from_torch_module, from_torch_tensor

from ..notification import Notification
from ..trainer import ModelTrainer, process_trainer_task
from ..trainer.events import SWAGRoundFinished, TrainingRoundFinished
from ..trainer.options import TrainerOptions

from .test_aggregation import _create_torchscript_model_and_init


class AiWorkerTest(TestCase):

    def test_process_training_good(self):
        base_model = Dummy.create_model(weights=from_torch_module(_create_torchscript_model_and_init(0)))
        training = Dummy.create_training(state=TrainingState.ONGOING, locked=True, model=base_model)
        model1 = _create_torchscript_model_and_init(0)
        model2 = _create_torchscript_model_and_init(1)

        Dummy.create_model_update(
            base_model=training.model,
            owner=training.participants.all()[0],
            round=0,
            weights=from_torch_module(model1),  # torchscript model
        )
        Dummy.create_model_update(
            base_model=training.model,
            owner=training.participants.all()[1],
            round=0,
            weights=from_torch_module(model2)  # torchscript model
        )
        with self.assertLogs(level="INFO"):
            ModelTrainer(training, TrainerOptions(skip_model_tests=True)).handle_cls(TrainingRoundFinished)
        res = training.model.get_torch_model().state_dict()
        torch.testing.assert_close(res["0.weight"], torch.ones_like(res["0.weight"]) * 0.5)
        torch.testing.assert_close(res["3.weight"], torch.ones_like(res["3.weight"]) * 0.5)

    @patch("fl_server_ai.aggregation.mean.MeanAggregation.aggregate")
    def test_process_training_bad1(self, aggregate_updates: MagicMock):
        base_model = Dummy.create_model(weights=from_torch_module(_create_torchscript_model_and_init(0)))
        training = Dummy.create_training(state=TrainingState.ONGOING, locked=True, model=base_model)
        model1 = _create_torchscript_model_and_init(0)
        model2 = _create_torchscript_model_and_init(1)
        model3 = _create_torchscript_model_and_init(500)
        Dummy.create_model_update(
            base_model=training.model,
            owner=training.participants.all()[0],
            round=0,
            weights=from_torch_module(model1),  # torchscript model
        )
        Dummy.create_model_update(
            base_model=training.model,
            owner=training.participants.all()[1],
            round=0,
            weights=from_torch_module(model2)  # torchscript model
        )
        Dummy.create_model_update(
            base_model=training.model,
            owner=training.participants.all()[1],
            round=0,
            weights=from_torch_module(model3)  # torchscript model
        )
        with self.assertLogs(level="ERROR"):
            with self.assertRaises(RuntimeError):
                ModelTrainer(training, TrainerOptions(skip_model_tests=True)).handle_cls(TrainingRoundFinished)

        self.assertFalse(aggregate_updates.called)

    @patch("fl_server_ai.aggregation.mean.MeanAggregation.aggregate")
    def test_process_training_bad2(self, aggregate_updates: MagicMock):
        base_model = Dummy.create_model(weights=from_torch_module(_create_torchscript_model_and_init(0)))
        training = Dummy.create_training(state=TrainingState.ONGOING, locked=True, model=base_model)
        model1 = _create_torchscript_model_and_init(0)
        Dummy.create_model_update(
            base_model=training.model,
            owner=training.participants.all()[0],
            round=0,
            weights=from_torch_module(model1),  # torchscript model
        )
        with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError):
            ModelTrainer(training, TrainerOptions(skip_model_tests=True)).handle_cls(TrainingRoundFinished)

        self.assertFalse(aggregate_updates.called)

    @patch.object(Notification, "send")
    def test_process_training(self, send_notification):
        model = Dummy.create_model(
            model_cls=SWAGModel,
            weights=from_torch_module(torch.jit.script(torch.nn.Sequential(
                # this model has exactly 15 parameters
                torch.nn.Linear(1, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            ))),
            round=100
        )
        training = Dummy.create_training(
            state=TrainingState.SWAG_ROUND,
            uncertainty_method=UncertaintyMethod.SWAG,
            locked=True,
            model=model,
            target_num_updates=100
        )
        Dummy.create_metric(
            step=100,
            key="SWAG First Moment Local",
            model=model,
            reporter=training.participants.all()[0],
            value_binary=from_torch_tensor(torch.zeros(15))
        )
        Dummy.create_metric(
            step=100,
            key="SWAG Second Moment Local",
            model=model,
            reporter=training.participants.all()[0],
            value_binary=from_torch_tensor(torch.ones(15))
        )
        Dummy.create_metric(
            step=100,
            key="SWAG Sample Size Local",
            model=model,
            reporter=training.participants.all()[0],
            value_float=1000
        )
        Dummy.create_metric(
            step=100,
            key="SWAG First Moment Local",
            model=model,
            reporter=training.participants.all()[1],
            value_binary=from_torch_tensor(torch.zeros(15))
        )
        Dummy.create_metric(
            step=100,
            key="SWAG Second Moment Local",
            model=model,
            reporter=training.participants.all()[1],
            value_binary=from_torch_tensor(torch.ones(15))
        )
        Dummy.create_metric(
            step=100,
            key="SWAG Sample Size Local",
            model=model,
            reporter=training.participants.all()[1],
            value_float=1000
        )
        with self.assertLogs("fl.server", level="INFO") as cm:
            process_trainer_task(training.id, SWAGRoundFinished)
        self.assertEqual(cm.output, [
            f"INFO:fl.server:Training {training.id}: Doing SWAG aggregation as all 2 updates arrived",
            f"INFO:fl.server:SWAG completed for training {training.id}",
        ])
        self.assertTrue(send_notification.called)
        training = Training.objects.get(id=training.id)
        self.assertFalse(training.locked)
        model = training.model
        self.assertEquals(TrainingState.ONGOING, training.state)  # next would be ModelTestFinished
        fst = model.first_moment
        snd = model.second_moment
        torch.testing.assert_close(torch.zeros(15), fst)
        torch.testing.assert_close(torch.ones(15), snd)
