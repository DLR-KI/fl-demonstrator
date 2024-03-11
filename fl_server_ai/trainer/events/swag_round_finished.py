from django.db.models.query import QuerySet
import torch
from typing import List

from fl_server_core.models import Metric
from fl_server_core.models.training import TrainingState

from ...exceptions import AggregationException
from .training_round_finished import TrainingRoundFinished


class SWAGRoundFinished(TrainingRoundFinished):

    def next(self):
        self.training.state = TrainingState.ONGOING
        self.training.save()
        super().next()

    def handle(self):
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
        if not all(map(lambda v: len(v) == len(param_vectors[0]), param_vectors[1:])):
            raise AggregationException("Models do not have the same number of parameters!")
        if len(param_vectors) != len(sample_sizes):
            raise RuntimeError("len(sample_sizes) != len(param_vectors)")

        factors = torch.tensor([s / sum(sample_sizes) for s in sample_sizes])
        result = torch.stack(param_vectors) * factors[:, None]
        result = torch.sum(result, dim=0)
        return result
