# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TransactionTestCase
import json

from fl_server_core.models.training import Training, TrainingState
from fl_server_core.tests import BASE_URL, Dummy

from ..trainer import ModelTrainer, TrainerOptions
from ..trainer.events import TrainingRoundFinished


class AggregationProcessTest(TransactionTestCase):

    def setUp(self):
        self.user = Dummy.create_user_and_authenticate(self.client)

    def test_check_and_run_aggregation_if_applicable_training_step(self):
        clients = [Dummy.create_user() for _ in range(10)]
        training = Dummy.create_training(state=TrainingState.ONGOING, target_num_updates=10, actor=self.user)
        training.participants.set(clients)
        updates = [Dummy.create_model_update(base_model=training.model, round=0) for _ in range(10)]
        for update in updates:
            update.save()

        with self.assertLogs("fl.server", level="INFO"):
            ModelTrainer(training, TrainerOptions(skip_model_tests=True)).handle_cls(TrainingRoundFinished)

        response = self.client.get(f"{BASE_URL}/models/{training.model.id}/metadata/")
        content = json.loads(response.content)
        self.assertEqual(content["round"], 1)

        training = Training.objects.get(id=training.id)
        self.assertEqual(TrainingState.ONGOING, training.state)

    def test_check_and_run_aggregation_if_applicable_training_finished(self):
        clients = [Dummy.create_user() for _ in range(10)]
        model = Dummy.create_model(owner=self.user, round=1)
        training = Dummy.create_training(state=TrainingState.ONGOING, target_num_updates=1, actor=self.user,
                                         model=model)
        training.participants.set(clients)
        updates = [Dummy.create_model_update(base_model=training.model, round=1) for _ in range(10)]
        for update in updates:
            update.save()

        options = TrainerOptions(skip_model_tests=True)
        trainer = ModelTrainer(training, options)
        with self.assertLogs("fl.server", level="INFO"):
            trainer.handle_cls(TrainingRoundFinished)

        training = Training.objects.get(id=training.id)
        self.assertEqual(TrainingState.COMPLETED, training.state)
