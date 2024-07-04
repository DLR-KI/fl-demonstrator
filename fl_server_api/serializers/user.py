# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from rest_framework import serializers
from rest_framework.authtoken.models import Token

from fl_server_core.models import User


class UserSerializer(serializers.ModelSerializer):
    """
    A serializer for the User model.

    This serializer includes a method field for the user's token,
    which is only included in the serialized data if the
    request user is the same as the requested user.
    """

    token = serializers.SerializerMethodField()
    """A method field for the user's token."""

    class Meta:
        model = User
        fields = [
            "username", "first_name", "last_name",
            "email", "id", "actor", "client",
            "message_endpoint", "token", "password",
        ]
        extra_kwargs = {
            "id": {"read_only": True},
            "token": {"read_only": True},
            "password": {"write_only": True},
        }

    def get_token(self, user: User) -> str | None:
        """
        Get the user's token.

        The token is only returned if the request user is the same as the requested user.
        The request user ID is passed in the context.

        Args:
            user (User): The user instance.

        Returns:
            str | None: The user's token, or "**********" if the request user is not the same as the requested user.
        """
        if self.context.get("request_user_id") == user.id:
            return Token.objects.get(user=user).key
        return "**********"

    def to_representation(self, instance):
        """
        Generate a dictionary representation of the User instance.

        The token key is removed from the response if the request user is not the same as the requested user.

        Args:
            instance (User): The User instance.

        Returns:
            dict: The dictionary representation of the User instance.
        """
        # remove the token key from the response if the request user is not the same as
        # the requested user since its always empty or "**********"
        data = super().to_representation(instance)
        if data.get("token") == "**********":
            del data["token"]
        return data

    def create(self, validated_data):
        """
        Create a new User instance.

        The user's password is set using the `set_password` method.

        Args:
            validated_data (dict): The validated data for the new User instance.

        Returns:
            User: The created User instance.
        """
        user = User.objects.create(**validated_data)
        user.set_password(validated_data["password"])
        user.save()
        return user
