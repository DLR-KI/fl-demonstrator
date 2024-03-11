from dataclasses import dataclass, field
from marshmallow import fields, post_load, Schema
from typing import Any, Dict, List
from uuid import UUID

from fl_server_core.models.training import AggregationMethod, UncertaintyMethod


class CreateTrainingRequestSchema(Schema):
    model_id = fields.UUID()
    target_num_updates = fields.Integer()
    metric_names = fields.List(fields.Str())
    aggregation_method = fields.Enum(
        AggregationMethod,
        required=True,
        dump_default=AggregationMethod.FED_AVG,
        by_value=True
    )
    uncertainty_method = fields.Enum(
        UncertaintyMethod,
        required=False,
        dump_default=UncertaintyMethod.NONE,
        by_value=True
    )
    options = fields.Dict(required=False, dump_default={})
    clients = fields.List(fields.UUID(), dump_default=[])

    @post_load
    def _make_create_training_request(self, data: Dict[str, Any], **kwargs):
        return CreateTrainingRequest(**data)


@dataclass
class CreateTrainingRequest:
    model_id: UUID
    target_num_updates: int
    metric_names: list[str]
    aggregation_method: AggregationMethod = field(default=AggregationMethod.FED_AVG)  # type: ignore[assignment]
    uncertainty_method: UncertaintyMethod = field(default=UncertaintyMethod.NONE)  # type: ignore[assignment]
    options: Dict[str, Any] = field(default_factory=lambda: {})
    clients: List[UUID] = field(default_factory=lambda: [])


class ClientAdministrationBodySchema(Schema):
    clients = fields.List(fields.UUID, required=True)

    @post_load
    def _make_client_administration_body(self, data: Dict[str, Any], **kwargs):
        return ClientAdministrationBody(**data)


@dataclass
class ClientAdministrationBody:
    clients: list[UUID]
