from typing import overload, Type

from fl_server_core.models import Model, Training
from fl_server_core.models.training import UncertaintyMethod

from .base import UncertaintyBase
from .ensemble import Ensemble
from .mc_dropout import MCDropout
from .none import NoneUncertainty
from .swag import SWAG


@overload
def get_uncertainty_class(value: Model) -> Type[UncertaintyBase]: ...


@overload
def get_uncertainty_class(value: Training) -> Type[UncertaintyBase]: ...


@overload
def get_uncertainty_class(value: UncertaintyMethod) -> Type[UncertaintyBase]: ...


def get_uncertainty_class(value: Model | Training | UncertaintyMethod) -> Type[UncertaintyBase]:
    if isinstance(value, UncertaintyMethod):
        method = value
    elif isinstance(value, Training):
        method = value.uncertainty_method
    elif isinstance(value, Model):
        uncertainty_method = Training.objects.filter(model=value) \
                .values("uncertainty_method") \
                .first()["uncertainty_method"]
        method = uncertainty_method
    else:
        raise ValueError(f"Unknown type: {type(value)}")

    match method:
        case UncertaintyMethod.ENSEMBLE: return Ensemble
        case UncertaintyMethod.MC_DROPOUT: return MCDropout
        case UncertaintyMethod.NONE: return NoneUncertainty
        case UncertaintyMethod.SWAG: return SWAG
        case _: raise ValueError(f"Unknown uncertainty method: {method}")
