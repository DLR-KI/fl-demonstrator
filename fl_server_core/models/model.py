from django.db.models import (
    BinaryField, CASCADE, CharField, ForeignKey, IntegerField, ManyToManyField, TextField, UUIDField
)
from polymorphic.models import PolymorphicModel
import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional, Sequence, TypeVar
from uuid import uuid4
from copy import copy

from ..utils.torch_pickle import (
    from_torch_module, to_torch_module,
    from_torch_tensor, to_torch_tensor
)

from .. import models as models
from .user import User


class Model(PolymorphicModel):
    id: UUIDField = UUIDField(primary_key=True, editable=False, default=uuid4)
    owner: ForeignKey = ForeignKey(User, on_delete=CASCADE)
    round: IntegerField = IntegerField()
    weights: BinaryField = BinaryField()

    def is_global_model(self):
        return isinstance(self, GlobalModel)

    def is_local_model(self):
        return isinstance(self, LocalModel)

    def get_torch_model(self) -> Module:
        return to_torch_module(self.weights)

    def set_torch_model(self, value: Module):
        self.weights = from_torch_module(value)

    def get_training(self) -> Optional["models.Training"]:
        return models.Training.objects.filter(model=self).first()


class GlobalModel(Model):
    name: CharField = CharField(max_length=256)
    description: TextField = TextField()
    input_shape: CharField = CharField(max_length=100, null=True)


class SWAGModel(GlobalModel):
    swag_first_moment: BinaryField = BinaryField()
    swag_second_moment: BinaryField = BinaryField()

    @property
    def first_moment(self) -> Tensor:
        return to_torch_tensor(self.swag_first_moment)

    @first_moment.setter
    def first_moment(self, value: Tensor):
        self.swag_first_moment = from_torch_tensor(value)

    @property
    def second_moment(self) -> Tensor:
        return to_torch_tensor(self.swag_second_moment)

    @second_moment.setter
    def second_moment(self, value: Tensor):
        self.swag_second_moment = from_torch_tensor(value)


class MeanModel(GlobalModel):
    models: ManyToManyField = ManyToManyField(GlobalModel, related_name="mean_models")

    def get_torch_model(self) -> Module:
        return MeanModule([model.get_torch_model() for model in self.models.all()])

    def set_torch_model(self, value: Module):
        raise NotImplementedError()


class LocalModel(Model):
    base_model: ForeignKey = ForeignKey(GlobalModel, on_delete=CASCADE)
    sample_size: IntegerField = IntegerField()

    def get_training(self) -> Optional["models.Training"]:
        return models.Training.objects.filter(model=self.base_model).first()


TModel = TypeVar("TModel", bound=Model)


class MeanModule(Module):

    def __init__(self, models: Sequence[Model]):
        super().__init__()
        self.models = models

    def forward(self, input: Tensor) -> Tensor:
        return torch.stack([model(input) for model in self.models], dim=0).mean(dim=0)


def clone_model(model: Model) -> Model:
    """
    Copies a model instance in the database
    See https://docs.djangoproject.com/en/5.0/topics/db/queries/#copying-model-instances
    and stackoverflow.com/questions/4733609/how-do-i-clone-a-django-model-instance-object-and-save-it-to-the-database

    Args:
        model (Model):  the model to be copied

    Returns:
        Model: New Model instance that is a copy of the old one
    """
    model.save()
    new_model = copy(model)
    new_model.pk = None
    new_model.id = None
    new_model._state.adding = True
    try:
        delattr(new_model, '_prefetched_objects_cache')
    except AttributeError:
        pass
    new_model.save()
    new_model.owner = model.owner
    new_model.save()
    return new_model
