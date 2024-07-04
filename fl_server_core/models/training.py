# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
from django.db.models import (
    CASCADE, CharField, ForeignKey, IntegerField, JSONField, ManyToManyField,
    OneToOneField, UUIDField, BooleanField
)
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _
from uuid import uuid4

from .model import Model
from .user import User


class TrainingState(models.TextChoices):
    """
    Training state choices for a Training.
    """
    INITIAL = "I", _("Initial")
    ONGOING = "O", _("Ongoing")
    COMPLETED = "C", _("Completed")
    ERROR = "E", _("Error")
    SWAG_ROUND = "S", _("SwagRound")


class AggregationMethod(models.TextChoices):
    """
    Aggregation method choices for a Training.
    """
    FED_AVG = "FedAvg", _("FedAvg")
    FED_DC = "FedDC", _("FedDC")
    FED_PROX = "FedProx", _("FedProx")


class UncertaintyMethod(models.TextChoices):
    """
    Uncertainty method choices for a Training.
    """
    NONE = "NONE", _("None")
    ENSEMBLE = "ENSEMBLE", _("Ensemble")
    MC_DROPOUT = "MC_DROPOUT", _("MC Dropout")
    SWAG = "SWAG", _("SWAG")


class Training(models.Model):
    """
    Training model class.
    """

    id: UUIDField = UUIDField(primary_key=True, editable=False, default=uuid4)
    """Unique identifier for the training."""
    model: OneToOneField = OneToOneField(Model, on_delete=CASCADE)
    """Model used in the training."""
    actor: ForeignKey = ForeignKey(User, on_delete=CASCADE, related_name="actors")
    """User who is the actor of the training."""
    participants: ManyToManyField = ManyToManyField(User)
    """Users who are the participants of the training."""
    state: CharField = CharField(max_length=1, choices=TrainingState.choices)
    """State of the training."""
    target_num_updates: IntegerField = IntegerField()
    """Target number of updates for the training."""
    last_update = models.DateTimeField(auto_now=True)
    """Time of the last update."""
    uncertainty_method: CharField = CharField(
        max_length=32, choices=UncertaintyMethod.choices, default=UncertaintyMethod.NONE
    )
    """Uncertainty method used in the training."""
    aggregation_method: CharField = CharField(
        max_length=32, choices=AggregationMethod.choices, default=AggregationMethod.FED_AVG
    )
    """Aggregation method used in the training."""
    # HINT: https://docs.djangoproject.com/en/4.2/topics/db/queries/#querying-jsonfield
    options: JSONField = JSONField(default=dict, encoder=DjangoJSONEncoder)
    """Options for the training."""
    locked: BooleanField = BooleanField(default=False)
    """Flag indicating whether the training is locked."""


@receiver(post_save, sender=Training)
def post_save_training(sender, instance=None, created=False, *args, **kwargs):
    """
    Ensure that the correct `target_num_updates` is set for every new training.

    This method is called after saving a training instance.
    It is used to set the `target_num_updates` to the correct value if this training instance is newly created
    and daisy chaining is enabled.

    Args:
        sender: The model class.
        instance (Training, optional): The actual instance being saved. Defaults to None.
        created (bool, optional): A boolean; True if a new record was created. Defaults to False.
        *args: Additional arguments.
        **kwargs: Arbitrary keyword arguments.
    """
    if not created:
        return
    daisy_chain_period = instance.options.get("daisy_chain_period", 0)
    if daisy_chain_period <= 0:
        return
    instance.target_num_updates = instance.target_num_updates * daisy_chain_period
    instance.save()
