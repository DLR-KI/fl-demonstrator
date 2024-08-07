# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fl_server_core.models import LocalModel

from ...aggregation import get_aggregation_class

from .base import ModelTrainerEvent
from .model_test_finished import ModelTestFinished


class TrainingRoundFinished(ModelTrainerEvent):
    """
    Training round finished event.

    This event should only be triggered when all model updates (local models)
    that are to participate in the aggregation have arrived.
    """

    def next(self):
        tests_enabled = not self.trainer.options.skip_model_tests
        if tests_enabled and self.trainer.options.model_test_after_each_round:
            self.trainer.test_round()
        elif tests_enabled and self.training.model.round >= self.training.target_num_updates:
            # at least test the final trained model
            self.trainer.test_round()
        else:
            ModelTestFinished(self.trainer).next()

    def handle(self):
        """
        Handle the training round finished event.

        - aggregate all model updates (local models) into a new global model
        - save the new global model into the database (i.e. updates/overwrites the weights field of the model)
        - increase the round field of the global model by 1
        - delete the model updates (local models) from the database, if the trainer options do not disagree

        Note: If not enough updates have arrived, the method does nothing.
        """
        model_updates = LocalModel.objects.filter(base_model=self.training.model, round=self.training.model.round)
        models = [m.get_torch_model() for m in model_updates]
        model_sample_sizes = [m.sample_size for m in model_updates]
        n_participants = self.training.participants.count()

        # validate
        self._validate(models, n_participants)
        self._logger.info(f"Training {self.training.id}: Doing aggregation as all {n_participants} updates arrived")

        # do aggregation
        aggregation_cls = get_aggregation_class(self.training)
        final_model = aggregation_cls().aggregate(
            models,
            model_sample_sizes,
            deepcopy=not self.trainer.options.delete_local_models_after_aggregation
        )

        # write the result back to database and update the trainings round
        self.training.model.set_torch_model(final_model)
        self.training.model.round += 1
        self.training.model.save()

        # clean local updates
        if self.trainer.options.delete_local_models_after_aggregation:
            model_updates.delete()

    def _validate(self, models: List, n_participants: int):
        """
        Validate the models and participant number for the training.

        This method checks if there are any models and if the number of models matches the number of participants.
        If any of these conditions are not met, an error is logged and a `RuntimeError` is raised.

        Args:
            models (List): The list of models for the training.
            n_participants (int): The number of participants in the training.

        Raises:
            RuntimeError: If there are no models or if the number of models does not match the number of participants.
        """
        if not models:
            text = f"Aggregation was run for training {self.training.id} but no model updates were in db!"
            self._logger.error(text)
            raise RuntimeError(text)

        if len(models) != n_participants:
            text = f"Aggregation was started, but training {self.training.id} has {len(models)} updates," \
                f"but {n_participants} clients!"
            self._logger.error(text)
            raise RuntimeError(text)
