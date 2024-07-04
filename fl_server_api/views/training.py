# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.http import HttpRequest, HttpResponse, JsonResponse
import json
from marshmallow import Schema
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from rest_framework import status
from rest_framework.exceptions import ParseError, PermissionDenied
from rest_framework.response import Response
from uuid import UUID

from fl_server_core.models import (
    Model as ModelDB,
    Training as TrainingDB,
    User as UserDB,
)
from fl_server_core.models.model import clone_model
from fl_server_core.models.training import TrainingState
from fl_server_ai.trainer import ModelTrainer

from .base import ViewSet
from ..utils import get_entity
from ..serializers.generic import TrainingSerializer, TrainingSerializerWithRounds
from ..serializers.training import (
    CreateTrainingRequest, CreateTrainingRequestSchema,
    ClientAdministrationBody, ClientAdministrationBodySchema
)
from drf_spectacular.utils import extend_schema, inline_serializer
from rest_framework.fields import UUIDField, CharField, IntegerField, ListField
from ..openapi import error_response_403
from ..serializers.generic import ErrorSerializer


class Training(ViewSet):
    """
    Training model ViewSet.

    This ViewSet is used to create and manage trainings.
    """

    serializer_class = TrainingSerializer
    """The serializer for the ViewSet."""

    def _check_user_permission_for_training(self, user: UserDB, training_id: UUID | str) -> TrainingDB:
        """
        Check if a user has permission for a training.

        This method checks if the user is the actor of the training or a participant in the training.

        Args:
            user (UserDB): The user.
            training_id (UUID | str): The ID of the training.

        Returns:
            TrainingDB: The training.
        """
        if isinstance(training_id, str):
            training_id = UUID(training_id)
        training = get_entity(TrainingDB, pk=training_id)
        if training.actor != user and user not in training.participants.all():
            raise PermissionDenied()
        return training

    def _get_clients_from_body(self, body_raw: bytes) -> list[UserDB]:
        """
        Get clients or participants from a request body.

        This method retrieves and loads all client data associated with the provided list of UUIDs contained
        within the request's clients field in the request body.

        Args:
            body_raw (bytes): The raw request body.

        Returns:
            list[UserDB]: The clients.
        """
        body: ClientAdministrationBody = self._load_marshmallow_request(ClientAdministrationBodySchema(), body_raw)
        return self._get_clients_from_uuid_list(body.clients)

    def _get_clients_from_uuid_list(self, uuids: list[UUID]) -> list[UserDB]:
        """
        Get clients from a list of UUIDs.

        This method gets the clients with the IDs in the list of UUIDs from the database.

        Args:
            uuids (list[UUID]): The list of UUIDs.

        Returns:
            list[UserDB]: The clients.
        """
        if uuids is None or len(uuids) == 0:
            return []
        # Note: filter "in" does not raise UserDB.DoesNotExist exceptions
        clients = UserDB.objects.filter(id__in=uuids)
        if len(clients) != len(uuids):
            raise ParseError("Not all provided users were found!")
        return clients

    def _load_marshmallow_request(self, schema: Schema, json_data: str | bytes | bytearray):
        """
        Load JSON data using from a request using a Marshmallow schema.

        Args:
            schema (Schema): The Marshmallow schema to use for loading the request.
            json_data (str | bytes | bytearray): The JSON data to load.

        Raises:
            ParseError: If a MarshmallowValidationError occurs.

        Returns:
            dict: The loaded data.
        """
        try:
            return schema.load(json.loads(json_data))  # should `schema.loads` be used instead?
        except MarshmallowValidationError as e:
            raise ParseError(e.messages) from e

    @extend_schema(responses={
        status.HTTP_200_OK: TrainingSerializer(many=True),
        status.HTTP_400_BAD_REQUEST: ErrorSerializer,
        status.HTTP_403_FORBIDDEN: error_response_403,
    })
    def get_trainings(self, request: HttpRequest) -> HttpResponse:
        """
        Get information about all owned trainings.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: list of training data as json response
        """
        trainings = TrainingDB.objects.filter(actor=request.user)
        serializer = TrainingSerializer(trainings, many=True)
        return Response(serializer.data)

    @extend_schema(responses={
        status.HTTP_200_OK: TrainingSerializerWithRounds,
        status.HTTP_400_BAD_REQUEST: ErrorSerializer,
        status.HTTP_403_FORBIDDEN: error_response_403,
    })
    def get_training(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Get information about the selected training.

        Args:
            request (HttpRequest):  request object
            id (str):  training uuid

        Returns:
            HttpResponse: training data as json response
        """
        train = self._check_user_permission_for_training(request.user, id)
        serializer = TrainingSerializerWithRounds(train)
        return Response(serializer.data)

    @extend_schema(
        request=inline_serializer("EmptyBodySerializer", fields={}),
        responses={
            status.HTTP_200_OK: inline_serializer(
                "StartTrainingSuccessSerializer",
                fields={
                    "detail": CharField(default="Training started!")
                }
            ),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def start_training(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Start a training process.

        This method checks if there are any participants registered for the training process.
        If there are participants, it checks if the training process is in the INITIAL state and starts the training
        session.

        Args:
            request (HttpRequest): The request object, which includes information about the user making the request.
            id (str): The UUID of the training process to start.

        Raises:
            ParseError: If there are no participants registered for the training process or if the training process
                is not in the INITIAL state.

        Returns:
            HttpResponse: A JSON response indicating that the training process has started.
        """
        training = self._check_user_permission_for_training(request.user, id)
        if training.participants.count() == 0:
            raise ParseError("At least one participant must be registered!")
        if training.state != TrainingState.INITIAL:
            raise ParseError(f"Training {training.id} is not in state INITIAL!")
        ModelTrainer(training).start()
        return JsonResponse({"detail": "Training started!"}, status=status.HTTP_202_ACCEPTED)

    @extend_schema(
        request=inline_serializer(
            "RegisterClientsSerializer",
            fields={
                "clients": ListField(child=UUIDField())
            }
        ),
        responses={
            status.HTTP_200_OK: inline_serializer(
                "RegisteredClientsSuccessSerializer",
                fields={
                    "detail": CharField(default="Users registered as participants!")
                }
            ),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def register_clients(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Register one or more clients for a training process.

        This method is designed to be called by a POST request with a JSON body of the form
        `{"clients": [<list of UUIDs>]}`.
        It adds these clients as participants of the training process.

        Note: This method should be called once before the training process is started.

        Args:
            request (HttpRequest): The request object.
            id (str): The UUID of the training process.

        Returns:
            HttpResponse: 202 Response if clients were registered, else corresponding error code.
        """
        train = self._check_user_permission_for_training(request.user, id)
        clients = self._get_clients_from_body(request.body)
        train.participants.add(*clients)
        return JsonResponse({"detail": "Users registered as participants!"}, status=status.HTTP_202_ACCEPTED)

    @extend_schema(responses={
        status.HTTP_200_OK: inline_serializer(
            "DeleteTrainingSuccessSerializer",
            fields={
                "detail": CharField(default="Training removed!")
            }
        ),
        status.HTTP_400_BAD_REQUEST: ErrorSerializer,
        status.HTTP_403_FORBIDDEN: error_response_403,
    })
    def remove_training(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Remove an existing training process.

        Args:
            request (HttpRequest):  request object
            id (str):  training uuid

        Returns:
            HttpResponse: 200 Response if training was removed, else corresponding error code
        """
        training = get_entity(TrainingDB, pk=id)
        if training.actor != request.user:
            raise PermissionDenied("You are not the owner the training.")
        training.delete()
        return JsonResponse({"detail": "Training removed!"})

    @extend_schema(
        request=inline_serializer(
            "RemoveClientsSerializer",
            fields={
                "clients": ListField(child=UUIDField())
            }
        ),
        responses={
            status.HTTP_200_OK: inline_serializer(
                "RemovedClientsSuccessSerializer",
                fields={
                    "detail": CharField(default="Users removed from training participants!")
                }
            ),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def remove_clients(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Remove one or more clients from a training process.

        This method is designed to modify an already existing training process.

        Args:
            request (HttpRequest): The request object.
            id (str): The UUID of the training process.

        Returns:
            HttpResponse: 200 Response if clients were removed, else corresponding error code.
        """
        train = self._check_user_permission_for_training(request.user, id)
        clients = self._get_clients_from_body(request.body)
        train.participants.remove(*clients)
        return JsonResponse({"detail": "Users removed from training participants!"})

    @extend_schema(
        request=inline_serializer(
            name="TrainingCreationSerializer",
            fields={
                "model_id": CharField(),
                "target_num_updates": IntegerField(),
                "metric_names": ListField(child=CharField()),
                "aggregation_method": CharField(),
                "clients": ListField(child=UUIDField())
            }
        ),
        responses={
            status.HTTP_200_OK: inline_serializer("TrainingCreatedSerializer", fields={
                "detail": CharField(default="Training created successfully!"),
                "training_id": UUIDField(),
            }),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def create_training(self, request: HttpRequest) -> HttpResponse:
        """
        Create a new training process.

        This method is designed to be called by a POST request according to the `CreateTrainingRequestSchema`.
        The request should include a model file (the initial model) as an attached FILE.

        Args:
            request (HttpRequest):  The request object.

        Returns:
            HttpResponse: 201 if training could be registered.
        """
        parsed_request: CreateTrainingRequest = self._load_marshmallow_request(
            CreateTrainingRequestSchema(),
            request.body.decode("utf-8")
        )
        model = get_entity(ModelDB, pk=parsed_request.model_id)
        if model.owner != request.user:
            raise PermissionDenied()
        if TrainingDB.objects.filter(model=model).exists():
            # the selected model is already referenced by another training, so we need to copy it
            model = clone_model(model)

        clients = self._get_clients_from_uuid_list(parsed_request.clients)
        train = TrainingDB.objects.create(
            model=model,
            actor=request.user,
            target_num_updates=parsed_request.target_num_updates,
            state=TrainingState.INITIAL,
            uncertainty_method=parsed_request.uncertainty_method.value,
            aggregation_method=parsed_request.aggregation_method.value,
            options=parsed_request.options
        )
        train.participants.add(*clients)
        return JsonResponse({
            "detail": "Training created successfully!",
            "training_id": train.id
        }, status=status.HTTP_201_CREATED)
