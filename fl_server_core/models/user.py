from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db.models import BooleanField, URLField, UUIDField
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from uuid import UUID, uuid4


class NotificationReceiver():
    id: UUID
    message_endpoint: str


class User(AbstractUser, NotificationReceiver):
    id: UUIDField = UUIDField(primary_key=True, editable=False, default=uuid4)
    actor: BooleanField = BooleanField(default=False)
    client: BooleanField = BooleanField(default=False)
    message_endpoint: URLField = URLField()


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)
