# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.db.models.query import QuerySet
import torch
from typing import List

from fl_server_core.models import Metric
from fl_server_core.models.training import TrainingState

from ...exceptions import AggregationException
from .training_round_finished import TrainingRoundFinished


class SWAGRoundFinished(TrainingRoundFinished):
    """
    Stochastic weight averaging Gaussian (SWAG) round finished event.
    """

    def next(self):
        self.training.state = TrainingState.ONGOING
        self.training.save()
        super().next()

    def handle(self):
        """
        Handle the SWAG event by collecting the SWAG first and second moments
        from all participants and aggregating them.
        """
        # collect metric value
        swag_fst = [m.to_torch() for m in self._get_metric("SWAG First Moment Local")]
        swag_snd = [m.to_torch() for m in self._get_metric("SWAG Second Moment Local")]
        sample_sizes = [m.value_float for m in self._get_metric("SWAG Sample Size Local")]
        n_participants = self.training.participants.count()

        # validate
        self._validate_swag(swag_fst, swag_snd, sample_sizes, n_participants)
        self._logger.info(
            f"Training {self.training.id}: Doing SWAG aggregation as all {n_participants} updates arrived"
        )

        # SWAG aggregation and save
        self.training.model.first_moment = self._aggregate_param_vectors(swag_fst, sample_sizes)
        self.training.model.second_moment = self._aggregate_param_vectors(swag_snd, sample_sizes)
        self.training.model.save()
        self._logger.info(f"SWAG completed for training {self.training.id}")

    def _get_metric(self, key: str) -> QuerySet[Metric]:
        """
        Get database metrics that match the training model and round as well as given key.

        Args:
            key (str): The key of the metric to retrieve.

        Returns:
            QuerySet[Metric]: A QuerySet of Metric objects that match the training model and round as well as given key.
        """
        return Metric.objects.filter(
            model=self.training.model,
            step=self.training.model.round,
            key=key
        )

    def _validate_swag(
        self,
        swag_fst: List[torch.Tensor],
        swag_snd: List[torch.Tensor],
        sample_sizes: List[int],
        n_participants: int
    ):
        """
        Validate the SWAG parameters and participant number for the training.

        This method checks if the lengths of first and second SWAG moments, sample sizes
        as well as the number of participants.
        If any of these conditions are not met, an error is logged and a `RuntimeError` is raised.

        Args:
            swag_fst (List[torch.Tensor]): List of first SWAG moments.
            swag_snd (List[torch.Tensor]): List of second SWAG moments.
            sample_sizes (List[int]): List of sample sizes.
            n_participants (int): The number of participants in the training.

        Raises:
            ValueError: If the lengths of first and second SWAG moments, and sample sizes are not equal.
            RuntimeError: If the length of first SWAG moments does not match the number of participants.
        """
        if len(swag_fst) != len(swag_snd) != len(sample_sizes):
            self.training.state = TrainingState.ERROR
            self.training.save()
            raise ValueError("SWAG stats in inconsistent state!")

        if len(swag_fst) != n_participants:
            text = f"Aggregation was started, but training {self.training.id}" \
                f"has {len(swag_fst)} updates," \
                f"but {n_participants} clients!"
            self._logger.error(text)
            raise RuntimeError(text)

    @torch.no_grad()
    def _aggregate_param_vectors(
        self,
        param_vectors: List[torch.Tensor],
        sample_sizes: List[int]
    ) -> torch.Tensor:
        """
        Aggregate parameter vectors using sample sizes.

        This method checks if all parameter vectors have the same length and if the length of
        parameter vectors matches the length of sample sizes.
        If any of these conditions are not met, a `RuntimeError` is raised.

        Args:
            param_vectors (List[torch.Tensor]): List of parameter vectors.
            sample_sizes (List[int]): List of sample sizes.

        Returns:
            torch.Tensor: Aggregated parameter vector.

        Raises:
            AggregationException: If not all parameter vectors have the same length.
            RuntimeError: If the length of sample sizes does not match the length of parameter vectors.
        """
        if not all(map(lambda v: len(v) == len(param_vectors[0]), param_vectors[1:])):
            raise AggregationException("Models do not have the same number of parameters!")
        if len(param_vectors) != len(sample_sizes):
            raise RuntimeError("len(sample_sizes) != len(param_vectors)")

        factors = torch.tensor([s / sum(sample_sizes) for s in sample_sizes])
        result = torch.stack(param_vectors) * factors[:, None]
        result = torch.sum(result, dim=0)
        return result
