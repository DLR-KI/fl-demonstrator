# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from django.contrib.postgres.fields import ArrayField
from django.db.models import (
    BinaryField, CASCADE, CharField, ForeignKey, IntegerField, ManyToManyField, TextField, UUIDField
)
from polymorphic.models import PolymorphicModel
import torch
from torch import Tensor
from torch.nn import Module
from typing import List, Optional, Sequence, TypeVar
from uuid import uuid4

from ..utils.torch_serialization import (
    from_torch_module, to_torch_module,
    from_torch_tensor, to_torch_tensor,
    is_torchscript_instance
)

from .. import models as models
from .user import User


class Model(PolymorphicModel):
    """
    Base model class for all types of model models.
    """

    id: UUIDField = UUIDField(primary_key=True, editable=False, default=uuid4)
    """Unique identifier for the model."""
    owner: ForeignKey = ForeignKey(User, on_delete=CASCADE)
    """User who owns the model."""
    round: IntegerField = IntegerField()
    """Round number of the model."""
    weights: BinaryField = BinaryField()
    """Weights of the model."""

    def is_global_model(self):
        """
        Checks if the model is a global model.

        Returns:
            bool: True if the model is a global model, False otherwise.
        """
        return isinstance(self, GlobalModel)

    def is_local_model(self):
        """
        Checks if the model is a local model.

        Returns:
            bool: True if the model is a local model, False otherwise.
        """
        return isinstance(self, LocalModel)

    def get_torch_model(self) -> torch.nn.Module:
        """
        Converts the model weights to a PyTorch model.

        Returns:
            torch.nn.Module: The PyTorch model.
        """
        return to_torch_module(self.weights)

    def set_torch_model(self, value: torch.nn.Module):
        """
        Sets the model weights from a PyTorch model.

        Args:
            value (torch.nn.Module): The PyTorch model.
        """
        self.weights = from_torch_module(value)

    def get_training(self) -> Optional["models.Training"]:
        """
        Gets the training associated with the model.

        Returns:
            models.Training: The training associated with the model.
        """
        return models.Training.objects.filter(model=self).first()


class GlobalModel(Model):
    """
    Model class for global models.
    """

    name: CharField = CharField(max_length=256)
    """Name of the model."""
    description: TextField = TextField()
    """Description of the model."""
    # alternative to be postgres independent: create a new model for each nullable integer field
    # and map the corresponding list of integers to the model (but pay attention to the order)
    input_shape: ArrayField = ArrayField(IntegerField(null=True), null=True)
    """Input shape of the model."""
    preprocessing: BinaryField = BinaryField(null=True)
    """Preprocessing of the model."""

    def get_preprocessing_torch_model(self) -> torch.nn.Module:
        """
        Converts the preprocessing to a PyTorch model.

        Returns:
            torch.nn.Module: The PyTorch model.
        """
        return to_torch_module(self.preprocessing)

    def set_preprocessing_torch_model(self, value: torch.nn.Module):
        """
        Sets the preprocessing from a PyTorch model.

        Args:
            value (torch.nn.Module): The PyTorch model.
        """
        self.preprocessing = from_torch_module(value)


class SWAGModel(GlobalModel):
    """
    Model class for SWAG models.
    """

    swag_first_moment: BinaryField = BinaryField()
    """First moment of the SWAG model."""
    swag_second_moment: BinaryField = BinaryField()
    """Second moment of the SWAG model."""

    @property
    def first_moment(self) -> Tensor:
        """
        Gets the first moment of the SWAG model.

        Returns:
            Tensor: The first moment of the SWAG model.
        """
        return to_torch_tensor(self.swag_first_moment)

    @first_moment.setter
    def first_moment(self, value: Tensor):
        """
        Sets the first moment of the SWAG model.

        Args:
            value (Tensor): The first moment of the SWAG model.
        """
        self.swag_first_moment = from_torch_tensor(value)

    @property
    def second_moment(self) -> Tensor:
        """
        Gets the second moment of the SWAG model.

        Returns:
            Tensor: The second moment of the SWAG model.
        """
        return to_torch_tensor(self.swag_second_moment)

    @second_moment.setter
    def second_moment(self, value: Tensor):
        """
        Sets the second moment of the SWAG model.

        Args:
            value (Tensor): The second moment of the SWAG model.
        """
        self.swag_second_moment = from_torch_tensor(value)


class MeanModel(GlobalModel):
    """
    Model class for mean models.
    """

    models: ManyToManyField = ManyToManyField(GlobalModel, related_name="mean_models")
    """Models of the mean model."""

    def get_torch_model(self) -> torch.nn.Module:
        """
        Converts the models to a PyTorch model.

        Returns:
            torch.nn.Module: The PyTorch model.
        """
        torch_models: List[torch.nn.Module] = [model.get_torch_model() for model in self.models.all()]
        model = MeanModule(torch_models)
        if all(is_torchscript_instance(m) for m in torch_models):
            return torch.jit.script(model)
        return model

    def set_torch_model(self, value: torch.nn.Module):
        """
        Sets the models from a PyTorch model.

        Args:
            value (torch.nn.Module): The PyTorch model.
        """
        raise NotImplementedError()


class LocalModel(Model):
    """
    Model class for local models.
    """

    base_model: ForeignKey = ForeignKey(GlobalModel, on_delete=CASCADE)
    """Base model of the local model."""
    sample_size: IntegerField = IntegerField()
    """Sample size of the local model."""

    def get_training(self) -> Optional["models.Training"]:
        """
        Gets the training associated with the base model.

        Returns:
            models.Training: The training associated with the base model.
        """
        return models.Training.objects.filter(model=self.base_model).first()


TModel = TypeVar("TModel", bound=Model)


class MeanModule(Module):
    """
    PyTorch module for mean models.
    """

    def __init__(self, models: Sequence[torch.nn.Module]):
        """
        Initializes the mean models.

        Args:
            models (Sequence[torch.nn.Module]): The models of the mean model.
        """
        super().__init__()
        self.models = models
        """Models of the mean model."""

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the mean models.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
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
