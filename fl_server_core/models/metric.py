from django.db import models
from django.db.models import BinaryField, CASCADE, CharField, FloatField, ForeignKey, IntegerField
from torch import Tensor
from torch.nn import Module

from ..utils.torch_pickle import from_torch_module_or_tensor, to_torch_module_or_tensor

from .model import Model
from .user import User


class Metric(models.Model):
    model: ForeignKey = ForeignKey(Model, on_delete=CASCADE)
    identifier: CharField = CharField(max_length=64, null=True, blank=True)
    key: CharField = CharField(max_length=32)
    value_float: FloatField = FloatField(null=True, blank=True)
    value_binary: BinaryField = BinaryField(null=True, blank=True)
    step: IntegerField = IntegerField(null=True, blank=True)
    reporter: ForeignKey = ForeignKey(User, null=True, blank=True, on_delete=CASCADE)

    @property
    def value(self) -> float | bytes:
        if self.is_float():
            return self.value_float
        return self.value_binary

    @value.setter
    def value(self, value: float | int | bytes | Module | Tensor):
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
        self.value_float = None
        self.value_binary = None

    def is_float(self) -> bool:
        return self.value_float is not None

    def is_binary(self) -> bool:
        return self.value_binary is not None

    def to_torch(self) -> Module | Tensor:
        return to_torch_module_or_tensor(self.value_binary)
