# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class TrainerOptions:
    """
    Trainer options including their default values.
    """

    skip_model_tests: bool = False
    """Flag indicating if model tests should be skipped."""
    model_test_after_each_round: bool = True
    """Flag indicating if a model test should be performed after each round."""
    delete_local_models_after_aggregation: bool = True
    """Flag indicating if local models should be deleted after aggregation."""
