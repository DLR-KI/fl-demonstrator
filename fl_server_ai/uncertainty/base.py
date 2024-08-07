# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import json
from logging import getLogger
import numpy as np
import torch
from typing import Any, Dict, Tuple

from fl_server_core.models import Model, Training


class UncertaintyBase(ABC):
    """
    Abstract base class for uncertainty estimation.

    This class defines the interface for uncertainty estimation in federated learning.
    """

    _logger = getLogger("fl.server")
    """The private logger instance for the uncertainty estimation."""

    @classmethod
    @abstractmethod
    def prediction(cls, input: torch.Tensor, model: Model) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Make a prediction using the given input and model.

        Args:
            input (torch.Tensor): The input to make a prediction for.
            model (Model): The model to use for making the prediction.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The prediction and any associated uncertainty.
        """
        pass

    @classmethod
    def interpret(cls, outputs: torch.Tensor) -> Dict[str, Any]:
        """
        Interpret the different network (model) outputs and calculate the uncertainty.

        Args:
            outputs (torch.Tensor): multiple network (model) outputs (N, batch_size, n_classes)

        Return:
            Tuple[torch.Tensor, Dict[str, Any]]: inference and uncertainty
        """
        variance = outputs.var(dim=0)
        std = outputs.std(dim=0)
        if not (torch.all(outputs <= 1.) and torch.all(outputs >= 0.)):
            return dict(variance=variance, std=std)

        predictive_entropy = cls.predictive_entropy(outputs)
        expected_entropy = cls.expected_entropy(outputs)
        mutual_info = predictive_entropy - expected_entropy  # see cls.mutual_information
        return dict(
            variance=variance,
            std=std,
            predictive_entropy=predictive_entropy,
            expected_entropy=expected_entropy,
            mutual_info=mutual_info,
        )

    @classmethod
    def expected_entropy(cls, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mean entropy of the predictive distribution across the MC samples.

        Args:
            predictions (torch.Tensor): predictions of shape (n_mc x batch_size x n_classes)

        Returns:
            torch.Tensor: mean entropy of the predictive distribution
        """
        return torch.distributions.Categorical(probs=predictions).entropy().mean(dim=0)

    @classmethod
    def predictive_entropy(cls, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of the mean predictive distribution across the MC samples.

        Args:
            predictions (torch.Tensor): predictions of shape (n_mc x batch_size x n_classes)

        Returns:
            torch.Tensor: entropy of the mean predictive distribution
        """
        return torch.distributions.Categorical(probs=predictions.mean(dim=0)).entropy()

    @classmethod
    def mutual_information(cls, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the BALD (Bayesian Active Learning by Disagreement) of a model;
        the difference between the mean of the entropy and the entropy of the mean
        of the predicted distribution on the predictions.
        This method is also sometimes referred to as the mutual information (MI).

        Args:
            predictions (torch.Tensor): predictions of shape (n_mc x batch_size x n_classes)

        Returns:
            torch.Tensor: difference between the mean of the entropy and the entropy of the mean
                    of the predicted distribution
        """
        return cls.predictive_entropy(predictions) - cls.expected_entropy(predictions)

    @classmethod
    def get_options(cls, obj: Model | Training) -> Dict[str, Any]:
        """
        Get uncertainty options from training options.

        Args:
            obj (Model | Training): The Model or Training object to retrieve options for.

        Returns:
            Dict[str, Any]: Uncertainty options.

        Raises:
            TypeError: If the given object is not a Model or Training.
        """
        if isinstance(obj, Model):
            return Training.objects.filter(model=obj) \
                .values("options") \
                .first()["options"] \
                .get("uncertainty", {})
        if isinstance(obj, Training):
            return obj.options.get("uncertainty", {})
        raise TypeError(f"Expected Model or Training, got {type(obj)}")

    @classmethod
    def to_json(cls, inference: torch.Tensor, uncertainty: Dict[str, Any] = {}, **json_kwargs) -> str:
        """
        Convert the given inference and uncertainty data to a JSON string.

        Args:
            inference (torch.Tensor): The inference to convert.
            uncertainty (Dict[str, Any]): The uncertainty to convert.
            **json_kwargs: Additional keyword arguments to pass to `json.dumps`.

        Returns:
            str: A JSON string representation of the given inference and uncertainty data.
        """
        def prepare(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().tolist()
            if isinstance(v, np.ndarray):  # cspell:ignore ndarray
                return v.tolist()
            if isinstance(v, dict):
                return {k: prepare(_v) for k, _v in v.items()}
            return v

        return json.dumps({
            "inference": inference.tolist(),
            "uncertainty": prepare(uncertainty) if uncertainty else {},
        }, **json_kwargs)
