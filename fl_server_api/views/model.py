# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import AnonymousUser
from django.db import transaction
from django.db.models import Q
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.http.response import HttpResponseBase
from django.utils.datastructures import MultiValueDictKeyError
from drf_spectacular.utils import extend_schema, OpenApiExample, inline_serializer, OpenApiResponse
from itertools import groupby
from rest_framework import status
from rest_framework.exceptions import APIException, NotFound, ParseError, PermissionDenied, ValidationError
from rest_framework.response import Response
from rest_framework.fields import UUIDField, CharField
import torch
from typing import Any, List, Union
from uuid import UUID

from fl_server_core.models import (
    GlobalModel as GlobalModelDB,
    LocalModel as LocalModelDB,
    Metric as MetricDB,
    Model as ModelDB,
    SWAGModel as SWAGModelDB,
    User as UserDB,
)
from fl_server_core.models.training import Training, TrainingState
from fl_server_core.utils.locked_atomic_transaction import locked_atomic_transaction
from fl_server_ai.trainer.events import ModelTestFinished, SWAGRoundFinished, TrainingRoundFinished
from fl_server_ai.trainer.tasks import dispatch_trainer_task

from .base import ViewSet
from ..utils import get_entity, get_file
from ..serializers.generic import ErrorSerializer, MetricSerializer
from ..serializers.model import ModelSerializer, ModelSerializerNoWeightsWithStats, load_and_create_model_request, \
    GlobalModelSerializer, ModelSerializerNoWeights, verify_model_object
from ..openapi import error_response_403, error_response_404


class Model(ViewSet):
    """
    Model ViewSet.
    """

    serializer_class = GlobalModelSerializer
    """The serializer for the ViewSet."""

    def _get_user_related_global_models(self, user: Union[AbstractBaseUser, AnonymousUser]) -> List[ModelDB]:
        """
        Get global models related to a user.

        This method retrieves all global models where the user is the actor or a participant of.

        Args:
            user (Union[AbstractBaseUser, AnonymousUser]): The user.

        Returns:
            List[ModelDB]: The global models related to the user.
        """
        user_ids = Training.objects.filter(
            Q(actor=user) | Q(participants=user)
        ).distinct().values_list("model__id", flat=True)
        return ModelDB.objects.filter(Q(owner=user) | Q(id__in=user_ids)).distinct()

    def _get_local_models_for_global_model(self, global_model: GlobalModelDB) -> List[LocalModelDB]:
        """
        Get all local models that are based on the global model.

        Args:
            global_model (GlobalModelDB): The global model.

        Returns:
            List[LocalModelDB]: The local models for the global model.
        """
        return LocalModelDB.objects.filter(base_model=global_model).all()

    def _get_local_models_for_global_models(self, global_models: List[GlobalModelDB]) -> List[LocalModelDB]:
        """
        Get all local models that are based on any of the global models.

        Args:
            global_models (List[GlobalModelDB]): The global models.

        Returns:
            List[LocalModelDB]: The local models for the global models.
        """
        return LocalModelDB.objects.filter(base_model__in=global_models).all()

    def _filter_by_training(self, models: List[ModelDB], training_id: str) -> List[ModelDB]:
        """
        Filter a list of models by checking if they are associated with the training.

        Args:
            models (List[ModelDB]): The models to filter.
            training_id (str): The ID of the training.

        Returns:
            List[ModelDB]: The models associated with the training.
        """
        def associated_with_training(m: ModelDB) -> bool:
            training = m.get_training()
            if training is None:
                return False
            return training.pk == UUID(training_id)
        return list(filter(associated_with_training, models))

    @extend_schema(
        responses={
            status.HTTP_200_OK: ModelSerializerNoWeights(many=True),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_models(self, request: HttpRequest) -> HttpResponse:
        """
        Get a list of all global models associated with the requesting user.

        A global model is deemed associated with a user if the user is either the owner of the model,
        or if the user is an actor or a participant in the model's training process.

        Args:
            request (HttpRequest): The incoming request object.

        Returns:
            HttpResponse: Model list as JSON response.
        """
        models = self._get_user_related_global_models(request.user)
        serializer = ModelSerializer(models, many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: ModelSerializerNoWeights(many=True),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_training_models(self, request: HttpRequest, training_id: str) -> HttpResponse:
        """
        Get a list of all models associated with a specific training process and the requesting user.

        A model is deemed associated with a user if the user is either the owner of the model,
        or if the user is an actor or a participant in the model's training process.

        Args:
            request (HttpRequest): The incoming request object.
            training_id (str): The unique identifier of the training process.

        Returns:
            HttpResponse: Model list as JSON response.
        """
        global_models = self._get_user_related_global_models(request.user)
        global_models = self._filter_by_training(global_models, training_id)
        local_models = self._get_local_models_for_global_models(global_models)
        serializer = ModelSerializer([*global_models, *local_models], many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: ModelSerializerNoWeights(many=True),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_training_models_latest(self, request: HttpRequest, training_id: str) -> HttpResponse:
        """
        Get a list of the latest models for a specific training process associated with the requesting user.

        A model is considered associated with a user if the user is either the owner of the model,
        or if the user is an actor or a participant in the model's training process.
        The latest model refers to the model from the most recent round (highest round number) of
        a participant's training process.

        Args:
            request (HttpRequest): The incoming request object.
            training_id (str): The unique identifier of the training process.

        Returns:
            HttpResponse: Model list as JSON response.
        """
        models: List[ModelDB] = []
        # add latest global model
        global_models = self._get_user_related_global_models(request.user)
        global_models = self._filter_by_training(global_models, training_id)
        models.append(max(global_models, key=lambda m: m.round))
        # add latest local models
        local_models = self._get_local_models_for_global_models(global_models)
        local_models = sorted(local_models, key=lambda m: str(m.owner.pk))  # required for groupby
        for _, group in groupby(local_models, key=lambda m: str(m.owner.pk)):
            models.append(max(group, key=lambda m: m.round))
        serializer = ModelSerializer(models, many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: ModelSerializerNoWeightsWithStats(),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_metadata(self, _request: HttpRequest, id: str) -> HttpResponse:
        """
        Get model meta data.

        Args:
            request (HttpRequest): The incoming request object.
            id (str): The unique identifier of the model.

        Returns:
            HttpResponse: Model meta data as JSON response.
        """
        model = get_entity(ModelDB, pk=id)
        serializer = ModelSerializer(model, context={"with-stats": True})
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: OpenApiResponse(response=bytes, description="Model is returned as bytes"),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_model(self, _request: HttpRequest, id: str) -> HttpResponseBase:
        """
        Download the whole model as PyTorch serialized file.

        Args:
            request (HttpRequest): The incoming request object.
            id (str): The unique identifier of the model.

        Returns:
            HttpResponseBase: model as file response
        """
        model = get_entity(ModelDB, pk=id)
        if isinstance(model, SWAGModelDB) and model.swag_first_moment is not None:
            if model.swag_second_moment is None:
                raise APIException(f"Model {model.id} is in inconsistent state!")
            raise NotImplementedError(
                "SWAG models need to be returned in 3 parts: model architecture, first moment, second moment"
            )
        # NOTE: FileResponse does strange stuff with bytes
        #       and in case of sqlite the weights will be bytes and not a memoryview
        response = HttpResponse(model.weights, content_type="application/octet-stream")
        response["Content-Disposition"] = f'filename="model-{id}.pt"'
        return response

    @extend_schema(
        responses={
            status.HTTP_200_OK: OpenApiResponse(
                response=bytes,
                description="Proprecessing model is returned as bytes"
            ),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
            status.HTTP_404_NOT_FOUND: error_response_404,
        },
    )
    def get_model_proprecessing(self, _request: HttpRequest, id: str) -> HttpResponseBase:
        """
        Download the whole preprocessing model as PyTorch serialized file.

        Args:
            request (HttpRequest): The incoming request object.
            id (str): The unique identifier of the model.

        Returns:
            HttpResponseBase: proprecessing model as file response or 404 if proprecessing model not found
        """
        model = get_entity(ModelDB, pk=id)
        global_model: torch.nn.Module
        if isinstance(model, GlobalModelDB):
            global_model = model
        elif isinstance(model, LocalModelDB):
            global_model = model.base_model
        else:
            self._logger.error("Unknown model type. Not a GlobalModel and not a LocalModel.")
            raise ValidationError(f"Unknown model type. Model id: {id}")
        if global_model.preprocessing is None:
            raise NotFound(f"Model '{id}' has no preprocessing model defined.")
        # NOTE: FileResponse does strange stuff with bytes
        #       and in case of sqlite the weights will be bytes and not a memoryview
        response = HttpResponse(global_model.preprocessing, content_type="application/octet-stream")
        response["Content-Disposition"] = f'filename="model-{id}-proprecessing.pt"'
        return response

    @extend_schema(responses={
        status.HTTP_200_OK: inline_serializer(
            "DeleteModelSuccessSerializer",
            fields={
                "detail": CharField(default="Model removed!")
            }
        ),
        status.HTTP_400_BAD_REQUEST: ErrorSerializer,
        status.HTTP_403_FORBIDDEN: error_response_403,
    })
    def remove_model(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Remove an existing model.

        Args:
            request (HttpRequest): The incoming request object.
            id (str): The unique identifier of the model.

        Returns:
            HttpResponse: 200 Response if model was removed, else corresponding error code
        """
        model = get_entity(ModelDB, pk=id)
        if model.owner != request.user:
            training = model.get_training()
            if training is None or training.actor != request.user:
                raise PermissionDenied(
                    "You are neither the owner of the model nor the actor of the corresponding training."
                )
        model.delete()
        return JsonResponse({"detail": "Model removed!"})

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "model_file": {"type": "string", "format": "binary"},
                    "model_preprocessing_file": {"type": "string", "format": "binary", "required": "false"},
                },
            },
        },
        responses={
            status.HTTP_201_CREATED: inline_serializer("ModelUploadSerializer", fields={
                "detail": CharField(default="Model Upload Accepted"),
                "model_id": UUIDField(),
            }),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def create_model(self, request: HttpRequest) -> HttpResponse:
        """
        Upload a global model file.

        The model file should be a PyTorch serialized model.
        Providing the model via `torch.save` as well as in TorchScript format is supported.

        Args:
            request (HttpRequest): The incoming request object.

        Returns:
            HttpResponse: upload success message as json response
        """
        model = load_and_create_model_request(request)
        return JsonResponse({
            "detail": "Model Upload Accepted",
            "model_id": str(model.id),
        }, status=status.HTTP_201_CREATED)

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "model_preprocessing_file": {"type": "string", "format": "binary"},
                },
            },
        },
        responses={
            status.HTTP_202_ACCEPTED: inline_serializer("PreprocessingModelUploadSerializer", fields={
                "detail": CharField(default="Proprocessing Model Upload Accepted"),
                "model_id": UUIDField(),
            }),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def upload_model_preprocessing(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Upload a preprocessing model file for a global model.

        The preprocessing model file should be a PyTorch serialized model.
        Providing the model via `torch.save` as well as in TorchScript format is supported.

        ```python
        transforms = torch.nn.Sequential(
            torchvision.transforms.CenterCrop(10),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        ```

        Make sure to only use transformations that inherit from `torch.nn.Module`.
        It is advised to use the `torchvision.transforms.v2` module for common transformations.

        Please note that this function is still in the beta phase.

        Args:
            request (HttpRequest): request object
            id (str): global model UUID

        Raises:
            PermissionDenied: Unauthorized to upload preprocessing model for the specified model
            ValidationError: Preprocessing model is not a valid torch model

        Returns:
            HttpResponse: upload success message as json response
        """
        model = get_entity(GlobalModelDB, pk=id)
        if request.user.id != model.owner.id:
            raise PermissionDenied(f"You are not the owner of model {model.id}!")
        model.preprocessing = get_file(request, "model_preprocessing_file")
        verify_model_object(model.preprocessing, "preprocessing")
        model.save()
        return JsonResponse({
            "detail": "Proprocessing Model Upload Accepted",
        }, status=status.HTTP_202_ACCEPTED)

    @extend_schema(
        responses={
            status.HTTP_200_OK: MetricSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_model_metrics(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Reports all metrics for the selected model.

        Args:
            request (HttpRequest):  request object
            id (str):  model UUID

        Returns:
            HttpResponse: Metrics as JSON Array
        """
        model = get_entity(ModelDB, pk=id)
        metrics = MetricDB.objects.filter(model=model).all()
        return Response(MetricSerializer(metrics, many=True).data)

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "metric_names": {"type": "list"},
                    "metric_values": {"type": "list"},
                },
            },
        },
        responses={
            status.HTTP_200_OK: inline_serializer("MetricUploadResponseSerializer", fields={
                "detail": CharField(default="Model Metrics Upload Accepted"),
                "model_id": UUIDField(),
            }),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
        examples=[
            OpenApiExample("Example", value={
                "metric_names": ["accuracy", "training loss"],
                "metric_values": [0.6, 0.04]
            }, media_type="multipart/form-data")
        ]
    )
    def create_model_metrics(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Upload model metrics.

        Args:
            request (HttpRequest):  request object
            id (str):  model uuid

        Returns:
            HttpResponse: upload success message as json response
        """
        model = get_entity(ModelDB, pk=id)
        formdata = dict(request.POST)

        with locked_atomic_transaction(MetricDB):
            self._metric_upload(formdata, model, request.user)

            if isinstance(model, GlobalModelDB):
                n_metrics = MetricDB.objects.filter(model=model, step=model.round).distinct("reporter").count()
                training = model.get_training()
                if training:
                    if n_metrics == training.participants.count():
                        dispatch_trainer_task(training, ModelTestFinished, False)
                else:
                    self._logger.warning(f"Global model {id} is not connected to any training.")

        return JsonResponse({
            "detail": "Model Metrics Upload Accepted",
            "model_id": str(model.id),
        }, status=status.HTTP_201_CREATED)

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "round": {"type": "int"},
                    "sample_size": {"type": "int"},
                    "metric_names": {"type": "list[string]"},
                    "metric_values": {"type": "list[float]"},
                    "model_file": {"type": "string", "format": "binary"},
                },
            },
        },
        responses={
            status.HTTP_200_OK: ModelSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def create_local_model(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Upload a partial trained model file from client.

        Args:
            request (HttpRequest):  request object
            id (str):  model uuid of the model, which was used for training

        Returns:
            HttpResponse: upload success message as json response
        """
        try:
            formdata = dict(request.POST)
            (round_num,) = formdata["round"]
            (sample_size,) = formdata["sample_size"]
            round_num, sample_size = int(round_num), int(sample_size)
            client = request.user
            model_file = get_file(request, "model_file")
            global_model = get_entity(GlobalModelDB, pk=id)

            # ensure that a training process coresponding to the model exists, else the process will error out
            training = Training.objects.get(model=global_model)
            self._verify_valid_update(client, training, round_num, TrainingState.ONGOING)

            verify_model_object(model_file)
            local_model = LocalModelDB.objects.create(
                base_model=global_model, weights=model_file,
                round=round_num, owner=client, sample_size=sample_size
            )
            self._metric_upload(formdata, local_model, client, metrics_required=False)

            updates = LocalModelDB.objects.filter(base_model=global_model, round=round_num)
            if updates.count() == training.participants.count():
                dispatch_trainer_task(training, TrainingRoundFinished, True)

            return JsonResponse({"detail": "Model Update Accepted"}, status=status.HTTP_201_CREATED)
        except Training.DoesNotExist:
            raise NotFound(f"Model with ID {id} does not have a training process running")
        except (MultiValueDictKeyError, KeyError) as e:
            raise ParseError(e)

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "round": {"type": "int"},
                    "sample_size": {"type": "int"},
                    "first_moment_file": {"type": "string", "format": "binary"},
                    "second_moment_file": {"type": "string", "format": "binary"}
                },
            },
        },
        responses={
            status.HTTP_200_OK: inline_serializer("MetricUploadSerializer", fields={
                "detail": CharField(default="SWAg Statistics Accepted"),
            }),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def create_swag_stats(self, request: HttpRequest, id: str) -> HttpResponse:
        """
        Upload SWAG statistics.

        Args:
            request (HttpRequest): request object
            id (str): global model uuid

        Raises:
            APIException: internal server error
            NotFound: model not found
            ParseError: request data not valid

        Returns:
            HttpResponse: upload success message as json response
        """
        try:
            client = request.user
            formdata = dict(request.POST)
            (round_num,) = formdata["round"]
            (sample_size,) = formdata["sample_size"]
            round_num, sample_size = int(round_num), int(sample_size)
            fst_moment = get_file(request, "first_moment_file")
            snd_moment = get_file(request, "second_moment_file")
            model = get_entity(GlobalModelDB, pk=id)

            # ensure that a training process coresponding to the model exists, else the process will error out
            training = Training.objects.get(model=model)
            self._verify_valid_update(client, training, round_num, TrainingState.SWAG_ROUND)

            self._save_swag_stats(fst_moment, snd_moment, model, client, sample_size)

            swag_stats_first = MetricDB.objects.filter(model=model, step=model.round, key="SWAG First Moment Local")
            swag_stats_second = MetricDB.objects.filter(model=model, step=model.round, key="SWAG Second Moment Local")

            if swag_stats_first.count() != swag_stats_second.count():
                training.state = TrainingState.ERROR
                raise APIException("SWAG stats in inconsistent state!")
            if swag_stats_first.count() == training.participants.count():
                dispatch_trainer_task(training, SWAGRoundFinished, True)

            return JsonResponse({"detail": "SWAG Statistic Accepted"}, status=status.HTTP_201_CREATED)
        except Training.DoesNotExist:
            raise NotFound(f"Model with ID {id} does not have a training process running")
        except (MultiValueDictKeyError, KeyError) as e:
            raise ParseError(e)
        except Exception as e:
            raise APIException(e)

    @staticmethod
    def _save_swag_stats(
        fst_moment: bytes, snd_moment: bytes, model: GlobalModelDB, client: UserDB, sample_size: int
    ):
        """
        Save the first and second moments, and the sample size of the SWAG to the database.

        This function creates and saves three metrics for each round of the model:

        - the first moment,
        - the second moment, and
        - the sample size.

        These metrics are associated with the model, the round, and the client that reported them.

        Args:
            fst_moment (bytes): The first moment of the SWAG.
            snd_moment (bytes): The second moment of the SWAG.
            model (GlobalModelDB): The global model for which the metrics are being reported.
            client (UserDB): The client reporting the metrics.
            sample_size (int): The sample size of the SWAG.
        """
        MetricDB.objects.create(
            model=model,
            key="SWAG First Moment Local",
            value_binary=fst_moment,
            step=model.round,
            reporter=client
        ).save()
        MetricDB.objects.create(
            model=model,
            key="SWAG Second Moment Local",
            value_binary=snd_moment,
            step=model.round,
            reporter=client
        ).save()
        MetricDB.objects.create(
            model=model,
            key="SWAG Sample Size Local",
            value_float=sample_size,
            step=model.round,
            reporter=client
        ).save()

    @transaction.atomic()
    def _metric_upload(self, formdata: dict, model: ModelDB, client: UserDB, metrics_required: bool = True):
        """
        Uploads metrics associated with a model.

        For each pair of metric name and value, it attempts to convert the value to a float.
        If this fails, it treats the value as a binary string.

        It then creates a new metric object with the model, the metric name, the float or binary value,
        the model's round number, and the client, and saves this object to the database.

        Args:
            formdata (dict): The form data containing the metric names and values.
            model (ModelDB): The model with which the metrics are associated.
            client (UserDB): The client reporting the metrics.
            metrics_required (bool): A flag indicating whether metrics are required. Defaults to True.

        Raises:
            ParseError: If `metric_names` or `metric_values` are not in formdata,
                or if they do not have the same length and metrics are required.
        """
        if "metric_names" not in formdata or "metric_values" not in formdata:
            if metrics_required or ("metric_names" in formdata) != ("metric_values" in formdata):
                raise ParseError("Metric names or values are missing")
            return
        if len(formdata["metric_names"]) != len(formdata["metric_values"]):
            if metrics_required:
                raise ParseError("Metric names and values must have the same length")
            return

        for key, value in zip(formdata["metric_names"], formdata["metric_values"]):
            try:
                metric_float = float(value)
                metric_binary = None
            except Exception:
                metric_float = None
                metric_binary = bytes(value, encoding="utf-8")
            MetricDB.objects.create(
                model=model,
                key=key,
                value_float=metric_float,
                value_binary=metric_binary,
                step=model.round,
                reporter=client
            ).save()

    def _verify_valid_update(self, client: UserDB, train: Training, round_num: int, expected_state: tuple[str, Any]):
        """
        Verifies if a client can update a training process.

        This function checks if

        - the client is a participant of the training process,
        - the training process is in the expected state, and if
        - the round number matches the current round of the model associated with the training process.

        Args:
            client (UserDB): The client attempting to update the training process.
            train (Training): The training process to be updated.
            round_num (int): The round number reported by the client.
            expected_state (tuple[str, Any]): The expected state of the training process.

        Raises:
            PermissionDenied: If the client is not a participant of the training process.
            ValidationError: If the training process is not in the expected state or if the round number does not match
                the current round of the model.
        """
        if client.id not in [p.id for p in train.participants.all()]:
            raise PermissionDenied(f"You are not a participant of training {train.id}!")
        if train.state != expected_state:
            raise ValidationError(f"Training with ID {train.id} is in state {train.state}")
        if int(round_num) != train.model.round:
            raise ValidationError(f"Training with ID {train.id} is not currently in round {round_num}")
