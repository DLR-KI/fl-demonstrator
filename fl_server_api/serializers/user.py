from rest_framework import serializers
from rest_framework.authtoken.models import Token

from fl_server_core.models import User


class UserSerializer(serializers.ModelSerializer):
    token = serializers.SerializerMethodField()

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
        # return the user token only if the request user is the same as the requested user
        if self.context.get("request_user_id") == user.id:
            return Token.objects.get(user=user).key
        return "**********"

    def to_representation(self, instance):
        # remove the token key from the response if the request user is not the same as
        # the requested user since its always empty or "**********"
        data = super().to_representation(instance)
        if data.get("token") == "**********":
            del data["token"]
        return data

    def create(self, validated_data):
        user = User.objects.create(**validated_data)
        user.set_password(validated_data["password"])
        user.save()
        return user
