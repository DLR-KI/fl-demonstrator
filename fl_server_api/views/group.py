from django.contrib.auth.models import Group as GroupModel
from django.http import HttpRequest, HttpResponse
from drf_spectacular.utils import extend_schema, OpenApiExample, OpenApiParameter
from rest_framework import status
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response

from fl_server_core.models import User as UserModel

from .base import ViewSet
from ..utils import get_entity
from ..serializers.generic import ErrorSerializer, GroupSerializer
from ..openapi import error_response_403


_default_group_example = OpenApiExample(
    name="Get group by id",
    description="\n\n".join([
        "Retrieve group data by group ID.",
        "_Please not that the user Jane Doe has to be created and authorized first._",
    ]),
    value=1,
    parameter_only=("id", OpenApiParameter.PATH)
)


class Group(ViewSet):

    serializer_class = GroupSerializer

    def _get_group(self, user: UserModel, group_id: int) -> GroupModel:
        """
        Get group by id if user is member of the group.
        Otherwise raise PermissionDenied.

        Args:
            user (UserModel):  user who makes the request
            group_id (int): group id

        Raises:
            PermissionDenied: If user is not member of the group.

        Returns:
            GroupModel: group instance
        """
        group = get_entity(GroupModel, pk=group_id)
        if not user.groups.contains(group):
            raise PermissionDenied("You are not allowed to access this group.")
        return group

    @extend_schema(
        responses={
            status.HTTP_200_OK: GroupSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def list(self, request: HttpRequest) -> HttpResponse:
        """
        Get all groups.

        Args:
            request (HttpRequest): request object

        Raises:
            PermissionDenied: If user is not a superuser.

        Returns:
            HttpResponse: list of groups as json response
        """
        if not request.user.is_superuser:
            raise PermissionDenied("You are not allowed to access all groups.")
        groups = GroupModel.objects.all()
        serializer = GroupSerializer(groups, many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: GroupSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[_default_group_example]
    )
    def retrieve(self, request: HttpRequest, id: int) -> HttpResponse:
        """
        Get group information by id.

        Args:
            request (HttpRequest): request object
            id (int): group id

        Returns:
            HttpResponse: group as json response
        """
        group = self._get_group(request.user, id)
        serializer = GroupSerializer(group)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_201_CREATED: GroupSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[OpenApiExample(
            name="Create group",
            description="Create a new group.",
            value={"name": "My new amazing group"},
        )]
    )
    def create(self, request: HttpRequest) -> HttpResponse:
        """
        Create a new group.

        Args:
            request (HttpRequest): request object

        Returns:
            HttpResponse: new created group as json response
        """
        serializer = GroupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def _update(self, request: HttpRequest, id: int, *, partial: bool) -> HttpResponse:
        """
        Update group information.

        Args:
            request (HttpRequest): request object
            id (int): group id
            partial (bool): allow partial update

        Returns:
            HttpResponse: updated group as json response
        """
        group = self._get_group(request.user, id)
        serializer = GroupSerializer(group, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        if getattr(group, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            group._prefetched_objects_cache = {}
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: GroupSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[
            _default_group_example,
            OpenApiExample(
                name="Update group",
                description="Update group fields.",
                value={"name": "My new amazing group is the best!"},
            )
        ]
    )
    def update(self, request: HttpRequest, id: int) -> HttpResponse:
        """
        Update group information.

        Args:
            request (HttpRequest): request object
            id (int): group id

        Returns:
            HttpResponse: updated group as json response
        """
        return self._update(request, id, partial=False)

    @extend_schema(
        responses={
            status.HTTP_200_OK: GroupSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[
            _default_group_example,
            OpenApiExample(
                name="Update group partially",
                description="Update only some group fields.",
                value={"name": "My new amazing group is the best!"},
            )
        ]
    )
    def partial_update(self, request: HttpRequest, id: int) -> HttpResponse:
        """
        Update group information partially.

        Args:
            request (HttpRequest): request object
            id (int): group id

        Returns:
            HttpResponse: updated group as json response
        """
        return self._update(request, id, partial=True)

    @extend_schema(
        responses={
            status.HTTP_204_NO_CONTENT: None,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[_default_group_example]
    )
    def destroy(self, request: HttpRequest, id: int) -> HttpResponse:
        """
        Remove group by id.

        Args:
            request (HttpRequest): request object
            id (int): group id

        Returns:
            HttpResponse: 204 NO CONTENT
        """
        group = self._get_group(request.user, id)
        group.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
