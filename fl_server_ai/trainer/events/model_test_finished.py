# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from .base import ModelTrainerEvent


class ModelTestFinished(ModelTrainerEvent):
    """
    Model test finished event.
    """

    def next(self):
        if self.training.model.round < self.training.target_num_updates:
            self.trainer.start_round()
        else:
            self.trainer.finish()

    def handle(self):
        # currently do nothing
        # Potentially, one could aggregate all common metrics here.
        pass
