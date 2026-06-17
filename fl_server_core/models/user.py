# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models import BooleanField, URLField, UUIDField
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.http import HttpRequest
from rest_framework.authtoken.models import Token
from uuid import UUID, uuid4


class NotificationReceiver():
    """
    Notification receiver base class.
    """

    id: UUID
    """Unique identifier for the notification receiver."""
    message_endpoint: str
    """Endpoint to send the message to."""


class User(AbstractUser, NotificationReceiver):
    """
    User class.

    Inherits from Django's AbstractUser and NotificationReceiver.
    """

    id: UUIDField = UUIDField(primary_key=True, editable=False, default=uuid4)
    """Unique identifier for the user."""
    actor: BooleanField = BooleanField(default=False)
    """Flag indicating whether the user is an actor."""
    client: BooleanField = BooleanField(default=False)
    """Flag indicating whether the user is a client."""
    message_endpoint: URLField = URLField()
    """Endpoint to send the message to."""


class Edc(models.Model):
    """
    EDC related configuration.
    """

    bpn = models.CharField(max_length=20, primary_key=True)
    """Business Partner Number (BPN)"""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    """User related to the EDC configuration."""


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, *args, **kwargs):
    """
    Ensure that an authentication token is created for every new user.

    Args:
        sender: The model class.
        instance (User, optional): The actual instance being saved. Defaults to None.
        created (bool, optional): A boolean; True if a new record was created. Defaults to False.
        *args: Additional arguments.
        **kwargs: Arbitrary keyword arguments.
    """
    if created:
        Token.objects.create(user=instance)


def create_edc_bpn(user: User, bpn: str | HttpRequest) -> Edc | None:
    """
    Save EDC BPN for user.

    Args:
        user (User): User.
        bpn (str | HttpRequest): EDC BPN or http request where the EDC BPN is included inside the header.

    Returns:
        Edc | None: EDC object or None if not successful.
    """
    if not isinstance(bpn, str):
        bpn = get_edc_bpn_from_request(bpn)
    if bpn is None:
        return None
    return Edc.objects.create(user=user, bpn=bpn)


def get_edc_bpn_from_request(request: HttpRequest) -> str | None:
    """
    Get EDC BPN from http request object (headers).

    Args:
        request (HttpRequest): http request object.

    Returns:
        str | None: EDC BPN or None if not found.
    """
    bpn = request.META.get("Edc-Bpn", "")
    contract_agreement_id = request.META.get("Edc-Contract-Agreement-Id", "")
    if not bpn or not contract_agreement_id:
        return None
    return bpn
