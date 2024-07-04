# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.contrib.auth.models import Group
from django.db import models
from rest_framework import serializers
from typing import List, Optional, Type

from fl_server_core.models import (
    Metric,
    Training,
    LocalModel,
)


class ErrorSerializer(serializers.Serializer):
    """
    Serializer for error messages.
    """

    detail = serializers.CharField()
    """Detailed error message."""


def _create_generic_serializer(
    cls: Type[models.Model],
    selected_fields: Optional[List[str]] = None
) -> Type[serializers.ModelSerializer]:
    """
    Create a generic database model serializer.

    Args:
        cls (Type[models.Model]): The database model to serialize.
        selected_fields (Optional[List[str]], optional): The fields of the model to serialize.
            If `None`, all fields of the model are serialized. Defaults to `None`.

    Returns:
        Type[serializers.ModelSerializer]: The serializer for the model.
    """
    class GenericSerializer(serializers.ModelSerializer):
        class Meta:
            model = cls
            fields = selected_fields if selected_fields else serializers.ALL_FIELDS
            extra_kwargs = {"id": {"read_only": True}}
    return type(f"{cls.__name__}Serializer", (GenericSerializer, ), {})


# Create generic serializers for database models.
TrainingSerializer: Type[serializers.ModelSerializer] = _create_generic_serializer(Training)
MetricSerializer: Type[serializers.ModelSerializer] = _create_generic_serializer(Metric)
GroupSerializer: Type[serializers.ModelSerializer] = _create_generic_serializer(Group, ["id", "name"])


class TrainingSerializerWithRounds(TrainingSerializer):
    """
    Serializer for the Training model including the update progress.
    """

    def to_representation(self, obj: Training):
        """
        Generate a dictionary representation of the Training model instance including the update progress.

        Args:
            obj (Training): The training model instance.

        Returns:
            dict: The dictionary representation of the training model instance.
        """
        base_representation = super().to_representation(obj)
        base_representation["local_models_arrived"] = len(LocalModel.objects.filter(base_model=obj.model))
        base_representation["local_models_expected"] = len(obj.participants.all())
        return base_representation
