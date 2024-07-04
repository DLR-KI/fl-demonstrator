# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from marshmallow import fields, post_load, Schema
from typing import Any, Dict, List
from uuid import UUID

from fl_server_core.models.training import AggregationMethod, UncertaintyMethod


class CreateTrainingRequestSchema(Schema):
    """
    A schema for serializing and deserializing `CreateTrainingRequest` instances.
    """

    model_id = fields.UUID()
    """ID of the model to be trained."""
    target_num_updates = fields.Integer()
    """Target number of updates for the training process."""
    metric_names = fields.List(fields.Str())
    """Names of the metrics to be used in the training process."""
    aggregation_method = fields.Enum(
        AggregationMethod,
        required=True,
        dump_default=AggregationMethod.FED_AVG,
        by_value=True
    )
    """Method to be used for aggregating updates. Defaults to `FED_AVG`."""
    uncertainty_method = fields.Enum(
        UncertaintyMethod,
        required=False,
        dump_default=UncertaintyMethod.NONE,
        by_value=True
    )
    """Method to be used for handling uncertainty. Defaults to `NONE`."""
    options = fields.Dict(required=False, dump_default={})
    """Additional options for the training process. Defaults to an empty dictionary."""
    clients = fields.List(fields.UUID(), dump_default=[])
    """Clients participating in the training process. Defaults to an empty list."""

    @post_load
    def _make_create_training_request(self, data: Dict[str, Any], **kwargs):
        """
        Create a `CreateTrainingRequest` instance from the loaded data.

        This method is called after the data has been loaded and validated.

        Args:
            data (Dict[str, Any]): The loaded data.
            **kwargs: Additional keyword arguments.

        Returns:
            CreateTrainingRequest: The created `CreateTrainingRequest` instance.
        """
        return CreateTrainingRequest(**data)


@dataclass
class CreateTrainingRequest:
    """
    A data class representing a request to create a training process.

    Attributes:
        model_id (UUID): The ID of the model to be trained.
        target_num_updates (int): The target number of updates for the training process.
        metric_names (list[str]): The names of the metrics to be used in the training process.
        aggregation_method (AggregationMethod): The method to be used for aggregating updates. Defaults to FED_AVG.
        uncertainty_method (UncertaintyMethod): The method to be used for handling uncertainty. Defaults to NONE.
        options (Dict[str, Any]): Additional options for the training process. Defaults to an empty dictionary.
        clients (List[UUID]): The clients participating in the training process. Defaults to an empty list.
    """
    model_id: UUID
    """ID of the model to be trained."""
    target_num_updates: int
    """Target number of updates for the training process."""
    metric_names: list[str]
    """Names of the metrics to be used in the training process."""
    aggregation_method: AggregationMethod = field(default=AggregationMethod.FED_AVG)  # type: ignore[assignment]
    """Method to be used for aggregating updates. Defaults to `FED_AVG`."""
    uncertainty_method: UncertaintyMethod = field(default=UncertaintyMethod.NONE)  # type: ignore[assignment]
    """Method to be used for handling uncertainty. Defaults to `NONE`."""
    options: Dict[str, Any] = field(default_factory=lambda: {})
    """Additional options for the training process. Defaults to an empty dictionary."""
    clients: List[UUID] = field(default_factory=lambda: [])
    """Clients participating in the training process. Defaults to an empty list."""


class ClientAdministrationBodySchema(Schema):
    """
    A schema for serializing and deserializing `ClientAdministrationBody` instances.
    """

    clients = fields.List(fields.UUID, required=True)
    """A list of UUIDs representing the clients to be administered."""

    @post_load
    def _make_client_administration_body(self, data: Dict[str, Any], **kwargs):
        """
        Create a `ClientAdministrationBody` instance from the loaded data.

        This method is called after the data has been loaded and validated.

        Args:
            data (Dict[str, Any]): The loaded data.
            **kwargs: Additional keyword arguments.

        Returns:
            ClientAdministrationBody: The created `ClientAdministrationBody` instance.
        """
        return ClientAdministrationBody(**data)


@dataclass
class ClientAdministrationBody:
    """
    A data class representing the body of a client administration request.
    """

    clients: List[UUID]
    """A list of UUIDs representing the clients to be administered."""
