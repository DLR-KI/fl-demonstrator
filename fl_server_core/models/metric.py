# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.db import models
from django.db.models import BinaryField, CASCADE, CharField, FloatField, ForeignKey, IntegerField
from torch import Tensor
from torch.nn import Module

from ..utils.torch_serialization import from_torch_module_or_tensor, to_torch_module_or_tensor

from .model import Model
from .user import User


class Metric(models.Model):
    """
    Metric model class.
    """

    model: ForeignKey = ForeignKey(Model, on_delete=CASCADE)
    """Model associated with the metric."""
    identifier: CharField = CharField(max_length=64, null=True, blank=True)
    """Identifier of the metric."""
    key: CharField = CharField(max_length=32)
    """Key of the metric."""
    value_float: FloatField = FloatField(null=True, blank=True)
    """Float value of the metric."""
    value_binary: BinaryField = BinaryField(null=True, blank=True)
    """Binary value of the metric."""
    step: IntegerField = IntegerField(null=True, blank=True)
    """Step of the metric."""
    reporter: ForeignKey = ForeignKey(User, null=True, blank=True, on_delete=CASCADE)
    """User who reported the metric."""

    @property
    def value(self) -> float | bytes:
        """
        Value of the metric.

        Returns:
            float | bytes: The value of the metric.
        """
        if self.is_float():
            return self.value_float
        return self.value_binary

    @value.setter
    def value(self, value: float | int | bytes | Module | Tensor):
        """
        Setter for the value of the metric.

        Args:
            value (float | int | bytes | Module | Tensor): The value to set.
        """
        if isinstance(value, float):
            self.value_float = value
        elif isinstance(value, int):
            self.value_float = float(value)
        elif isinstance(value, (Module, Tensor)):
            self.value_binary = from_torch_module_or_tensor(value)
        else:
            self.value_binary = value

    @value.deleter
    def value(self):
        """
        Deleter for the value of the metric.
        """
        self.value_float = None
        self.value_binary = None

    def is_float(self) -> bool:
        """
        Check if the value of the metric is a float.

        Returns:
            bool: `True` if the value of the metric is a float, otherwise `False`.
        """
        return self.value_float is not None

    def is_binary(self) -> bool:
        """
        Check if the value of the metric is binary.

        Returns:
            bool: `True` if the value of the metric is binary, otherwise `False`.
        """
        return self.value_binary is not None

    def to_torch(self) -> Module | Tensor:
        """
        Convert the binary value of the metric to a torch module or tensor.

        Returns:
            Module | Tensor: The converted torch module or tensor.
        """
        return to_torch_module_or_tensor(self.value_binary)
