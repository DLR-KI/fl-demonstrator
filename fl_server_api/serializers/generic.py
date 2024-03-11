from django.contrib.auth.models import Group
from django.db import models
from rest_framework import serializers
from typing import List, Optional, Type

from fl_server_core.models import (
    Metric,
    Training,
)


class ErrorSerializer(serializers.Serializer):
    detail = serializers.CharField()


def _create_generic_serializer(
    cls: Type[models.Model],
    selected_fields: Optional[List[str]] = None
) -> Type[serializers.ModelSerializer]:
    """
    Create a generic database model serializer.

    Args:
        cls (Type[models.Model]): database model
        selected_fields (Optional[List[str]], optional): model fields to serialize.
            If `None` serialize all model fields. Defaults to `None`.

    Returns:
        Type[serializers.ModelSerializer]: model serializer
    """
    class GenericSerializer(serializers.ModelSerializer):
        class Meta:
            model = cls
            fields = selected_fields if selected_fields else serializers.ALL_FIELDS
            extra_kwargs = {"id": {"read_only": True}}
    return type(f"{cls.__name__}Serializer", (GenericSerializer, ), {})


TrainingSerializer: Type[serializers.ModelSerializer] = _create_generic_serializer(Training)
MetricSerializer: Type[serializers.ModelSerializer] = _create_generic_serializer(Metric)
GroupSerializer: Type[serializers.ModelSerializer] = _create_generic_serializer(Group, ["id", "name"])
