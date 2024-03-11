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
from ..serializers.generic import TrainingSerializer
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
    Federated Learning Platform's API trainings endpoint `/api/trainings`.
    This endpoint is used to create and manage trainings.
    """

    serializer_class = TrainingSerializer

    def _check_user_permission_for_training(self, user: UserDB, training_id: UUID | str) -> TrainingDB:
        if isinstance(training_id, str):
            training_id = UUID(training_id)
        training = get_entity(TrainingDB, pk=training_id)
        if training.actor != user and user not in training.participants.all():
            raise PermissionDenied()
        return training

    def _get_clients_from_body(self, body_raw: bytes) -> list[UserDB]:
        body: ClientAdministrationBody = self._load_marshmallow_request(ClientAdministrationBodySchema(), body_raw)
        return self._get_clients_from_uuid_list(body.clients)

    def _get_clients_from_uuid_list(self, uuids: list[UUID]) -> list[UserDB]:
        if uuids is None or len(uuids) == 0:
            return []
        # Note: filter "in" does not raise UserDB.DoesNotExist exceptions
        clients = UserDB.objects.filter(id__in=uuids)
        if len(clients) != len(uuids):
            raise ParseError("Not all provided users were found!")
        return clients

    def _load_marshmallow_request(self, schema: Schema, json_data: str | bytes | bytearray):
        try:
            return schema.load(json.loads(json_data))  # should `schema.loads` be used instead?
        except MarshmallowValidationError as e:
            raise ParseError(e.messages)

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
        serializer = TrainingSerializer(train)
        return Response(serializer.data)

    @extend_schema(
        request=inline_serializer("EmptyBodySerializer", fields={}),
        responses={
            status.HTTP_200_OK: inline_serializer(
                "SuccessSerializer",
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
        Start training. Will check if at least one participant is registered.
        Should be called by an POST request with an empty body.

        Args:
            request (HttpRequest):  request object
            id (str):  training uuid

        Returns:
            HttpResponse: training data as json response
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
        Register one or more clients for the training.
        Should be called by POST with a json body of the form.
        Should be called once before the training is started.

        ```json
        {"clients": [<list of UUIDs>]}
        ```

        Will check if all the selected clients are registered.

        Args:
            request (HttpRequest):  request object
            id (str):  training uuid

        Returns:
            HttpResponse: 202 Response if clients were registered, else corresponding error code
        """
        train = self._check_user_permission_for_training(request.user, id)
        clients = self._get_clients_from_body(request.body)
        train.participants.add(*clients)
        return JsonResponse({"detail": "Users registered as participants!"}, status=status.HTTP_202_ACCEPTED)

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
        Remove clients from the list of registered clients for a training.
        This is meant for modification of an already existing training process.

        Args:
            request (HttpRequest):  request object
            id (str):  training uuid

        Returns:
            HttpResponse: 200 Response if clients were removed, else corresponding error code
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
                "model_id": UUIDField(),
            }),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        }
    )
    def create_training(self, request: HttpRequest) -> HttpResponse:
        """
        Creates a training process.
        Should be called by POST according to `CreateTrainingRequestSchema` and
        should have a model file (the initial model) as attached FILE.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: 201 if training could be registered
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
