# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
from unittest.mock import patch

from fl_server_core.tests.dummy import Dummy

from ..notification.training import TrainingRoundStartNotification
from ..trainer.events import DaisyChainRoundFinished, ModelTestFinished, TrainingRoundFinished
from ..trainer.model_trainer import get_trainer, get_trainer_class, FedDCModelTrainer


class FedDCTest(TestCase):

    def test_trainer_class(self):
        training = Dummy.create_training(options=dict(daisy_chain_period=5))
        trainer_cls = get_trainer_class(training)
        self.assertTrue(trainer_cls is FedDCModelTrainer)

    def test_trainer_type(self):
        training = Dummy.create_training(options=dict(daisy_chain_period=5))
        trainer = get_trainer(training)
        self.assertTrue(type(trainer) is FedDCModelTrainer)

    @patch.object(TrainingRoundStartNotification, "send")
    def test_start_dc_round(self, send_method):
        training = Dummy.create_training(options=dict(daisy_chain_period=5))
        trainer = get_trainer(training)
        trainer.start_round()
        self.assertTrue(send_method.called)
        send_method.assert_called_once()

    @patch.object(TrainingRoundStartNotification, "send")
    def test_start_permuted_dc_round(self, send_method):
        N = 10
        participants = [Dummy.create_client() for _ in range(N)]
        model = Dummy.create_model(round=2)
        [Dummy.create_model_update(base_model=model, owner=participant) for participant in participants]
        training = Dummy.create_training(
            model=model,
            participants=participants,
            target_num_updates=model.round - 1,
            options=dict(daisy_chain_period=5)
        )
        trainer = get_trainer(training)
        trainer.start_round()
        self.assertTrue(send_method.called)
        self.assertEqual(N, send_method.call_count)

    @patch.object(DaisyChainRoundFinished, "handle")
    @patch.object(DaisyChainRoundFinished, "next")
    def test_handle_round_finished(self, next_fn, handle_fn):
        training = Dummy.create_training(options=dict(daisy_chain_period=5))
        trainer = get_trainer(training)
        event = TrainingRoundFinished(trainer)
        trainer.handle(event)
        self.assertTrue(next_fn.called)
        next_fn.assert_called_once()
        self.assertTrue(handle_fn.called)
        handle_fn.assert_called_once()

    @patch.object(TrainingRoundFinished, "handle")
    @patch.object(DaisyChainRoundFinished, "next")
    def test_handle_round_finished_handle_dc_period(self, next_fn, base_handle_fn):
        training = Dummy.create_training(target_num_updates=42, options=dict(daisy_chain_period=5))
        trainer = get_trainer(training)
        event = TrainingRoundFinished(trainer)
        trainer.handle(event)
        self.assertTrue(next_fn.called)
        next_fn.assert_called_once()
        self.assertFalse(base_handle_fn.called)

    @patch.object(TrainingRoundFinished, "handle")
    @patch.object(DaisyChainRoundFinished, "next")
    def test_handle_round_finished_handle_trainings_epoch(self, next_fn, base_handle_fn):
        training = Dummy.create_training(options=dict(daisy_chain_period=5))
        trainer = get_trainer(training)
        event = TrainingRoundFinished(trainer)
        trainer.handle(event)
        self.assertTrue(next_fn.called)
        next_fn.assert_called_once()
        self.assertTrue(base_handle_fn.called)
        base_handle_fn.assert_called_once()

    @patch.object(ModelTestFinished, "handle")
    @patch.object(ModelTestFinished, "next")
    def test_handle_test_round_finished(self, next_fn, handle_fn):
        training = Dummy.create_training(options=dict(daisy_chain_period=5))
        trainer = get_trainer(training)
        event = ModelTestFinished(trainer)
        trainer.handle(event)
        self.assertTrue(next_fn.called)
        next_fn.assert_called_once()
        self.assertTrue(handle_fn.called)
        handle_fn.assert_called_once()
