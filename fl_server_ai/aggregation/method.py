from typing import overload, Type

from fl_server_core.models import Model, Training
from fl_server_core.models.training import AggregationMethod

from .base import Aggregation
from .fed_dc import FedDC
from .fed_prox import FedProx
from .mean import MeanAggregation


@overload
def get_aggregation_class(value: Model) -> Type[Aggregation]: ...


@overload
def get_aggregation_class(value: Training) -> Type[Aggregation]: ...


@overload
def get_aggregation_class(value: AggregationMethod) -> Type[Aggregation]: ...


def get_aggregation_class(value: AggregationMethod | Model | Training) -> Type[Aggregation]:
    if isinstance(value, AggregationMethod):
        method = value
    elif isinstance(value, Training):
        method = value.aggregation_method
    elif isinstance(value, Model):
        aggregation_method = Training.objects.filter(model=value) \
                .values("aggregation_method") \
                .first()["aggregation_method"]
        method = aggregation_method
    else:
        raise ValueError(f"Unknown type: {type(value)}")

    match method:
        case AggregationMethod.FED_AVG: return MeanAggregation
        case AggregationMethod.FED_DC: return FedDC
        case AggregationMethod.FED_PROX: return FedProx
        case _: raise ValueError(f"Unknown aggregation method: {method}")
