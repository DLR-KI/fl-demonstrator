# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
from typing import Sequence

from ..exceptions import AggregationException

from .base import Aggregation


class MeanAggregation(Aggregation):
    """
    Implements the aggregate method for aggregating models by calculating their mean.
    """

    @torch.no_grad()
    def aggregate(
        self,
        models: Sequence[torch.nn.Module],
        model_sample_sizes: Sequence[int],
        *,
        deepcopy: bool = True
    ) -> torch.nn.Module:
        """
        Aggregate models by calculating the mean.

        Args:
            models (Sequence[torch.nn.Module]): The models to be aggregated.
            model_sample_sizes (Sequence[int]): The sample sizes for each model.
            deepcopy (bool, optional): Whether to create a deep copy of the models. Defaults to True.

        Returns:
            torch.nn.Module: The aggregated model.

        Raises:
            AggregationException: If the models do not have the same architecture.
        """
        assert len(models) == len(model_sample_sizes)

        self._logger.debug(f"Doing mean aggregation for {len(models)} models!")
        model_state_dicts = [model.state_dict() for model in models]

        total_dataset_size = model_sample_sizes[0]
        result_dict = model_state_dicts[0]
        for layer_name in result_dict:
            result_dict[layer_name] *= model_sample_sizes[0]

        # sum accumulation
        for model_dict, dataset_size in zip(model_state_dicts[1:], model_sample_sizes[1:]):
            if set(model_dict.keys()) != set(result_dict.keys()):
                raise AggregationException("Models do not have the same architecture!")

            total_dataset_size += dataset_size
            for layer_name in result_dict:
                result_dict[layer_name] += model_dict[layer_name] * dataset_size

        # factor 1/n
        for layer_name in result_dict:
            result_dict[layer_name] = result_dict[layer_name] / total_dataset_size

        # return aggregated model
        result_model = copy.deepcopy(models[0]) if deepcopy else models[0]
        result_model.load_state_dict(result_dict)
        return result_model
