# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db.models import BooleanField, URLField, UUIDField
from django.db.models.signals import post_save
from django.dispatch import receiver
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
