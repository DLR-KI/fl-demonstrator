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
    INITIAL = "I", _("Initial")
    ONGOING = "O", _("Ongoing")
    COMPLETED = "C", _("Completed")
    ERROR = "E", _("Error")
    SWAG_ROUND = "S", _("SwagRound")


class AggregationMethod(models.TextChoices):
    FED_AVG = "FedAvg", _("FedAvg")
    FED_DC = "FedDC", _("FedDC")
    FED_PROX = "FedProx", _("FedProx")


class UncertaintyMethod(models.TextChoices):
    NONE = "NONE", _("None")
    ENSEMBLE = "ENSEMBLE", _("Ensemble")
    MC_DROPOUT = "MC_DROPOUT", _("MC Dropout")
    SWAG = "SWAG", _("SWAG")


class Training(models.Model):
    id: UUIDField = UUIDField(primary_key=True, editable=False, default=uuid4)
    model: OneToOneField = OneToOneField(Model, on_delete=CASCADE)
    actor: ForeignKey = ForeignKey(User, on_delete=CASCADE, related_name="actors")
    participants: ManyToManyField = ManyToManyField(User)
    state: CharField = CharField(max_length=1, choices=TrainingState.choices)
    target_num_updates: IntegerField = IntegerField()
    last_update = models.DateTimeField(auto_now=True)
    uncertainty_method: CharField = CharField(
        max_length=32, choices=UncertaintyMethod.choices, default=UncertaintyMethod.NONE
    )
    aggregation_method: CharField = CharField(
        max_length=32, choices=AggregationMethod.choices, default=AggregationMethod.FED_AVG
    )
    # HINT: https://docs.djangoproject.com/en/4.2/topics/db/queries/#querying-jsonfield
    options: JSONField = JSONField(default=dict, encoder=DjangoJSONEncoder)
    locked: BooleanField = BooleanField(default=False)


@receiver(post_save, sender=Training)
def post_save_training(sender, instance, created, *args, **kwargs):
    """
    Ensure the correct `target_num_updates` is set.

    This method is called after saving a training instance.
    It is used to set the `target_num_updates` to the correct value if this training instance is newly created
    and daisy chaining is enabled.
    """
    if not created:
        return
    daisy_chain_period = instance.options.get("daisy_chain_period", 0)
    if daisy_chain_period <= 0:
        return
    instance.target_num_updates = instance.target_num_updates * daisy_chain_period
    instance.save()
