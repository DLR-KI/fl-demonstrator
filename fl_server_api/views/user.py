from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from drf_spectacular.utils import extend_schema, OpenApiExample, OpenApiParameter
from rest_framework import decorators, status
from rest_framework.response import Response

from fl_server_core.models import User as UserModel, Training as TrainingModel

from .base import ViewSet
from ..utils import get_entity
from ..openapi import error_response_403
from ..serializers.generic import ErrorSerializer, GroupSerializer, TrainingSerializer
from ..serializers.user import UserSerializer


class User(ViewSet):

    serializer_class = UserSerializer

    @extend_schema(
        responses={
            status.HTTP_200_OK: UserSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def get_users(self, request: HttpRequest) -> HttpResponse:
        """
        Get current user information as list.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: user list as json response
        """
        serializer = UserSerializer([request.user], many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: UserSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[
            OpenApiExample(
                name="Get user by id",
                description="Retrieve user data by user ID.",
                value="88a9f11a-846b-43b5-bd15-367fc332ba59",
                parameter_only=("id", OpenApiParameter.PATH)
            )
        ]
    )
    def get_user(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Get user information.

        Args:
            request (HttpRequest):  request object
            id (str):  user uuid

        Returns:
            HttpResponse: user as json response
        """
        serializer = UserSerializer(get_entity(UserModel, pk=id), context={"request_user_id": request.user.id})
        return Response(serializer.data)

    #@decorators.authentication_classes([])
    #@decorators.permission_classes([])
    @extend_schema(
        responses={
            status.HTTP_200_OK: UserSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
        },
        examples=[
            OpenApiExample("Jane Doe", value={
                "message_endpoint": "http://example.com/",
                "actor": True,
                "client": True,
                "username": "jane",
                "first_name": "Jane",
                "last_name": "Doe",
                "email": "jane.doe@example.com",
                "password": "my-super-secret-password"
            })
        ]
    )
    def create_user(self, request: HttpRequest) -> HttpResponse:
        """
        Create a new user.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: new created user as json response
        """
        user = UserSerializer().create(request.data)
        serializer = UserSerializer(user, context={"request_user_id": user.id})
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(
        responses={
            status.HTTP_200_OK: GroupSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[
            OpenApiExample(
                name="User uuid",
                value="88a9f11a-846b-43b5-bd15-367fc332ba59",
                parameter_only=("id", OpenApiParameter.PATH)
            )
        ]
    )
    def get_user_groups(self, request: HttpRequest) -> HttpResponse:
        """
        Get user groups.

        Args:
            request (HttpRequest):  request object
            id (str):  user uuid

        Returns:
            HttpResponse: user groups as json response
        """
        serializer = GroupSerializer(request.user.groups, many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: TrainingSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def get_user_trainings(self, request: HttpRequest) -> HttpResponse:
        """
        Get user trainings.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: user trainings as json response
        """
        trainings = TrainingModel.objects.filter(Q(actor=request.user) | Q(participants=request.user)).distinct()
        serializer = TrainingSerializer(trainings, many=True)
        return Response(serializer.data)
