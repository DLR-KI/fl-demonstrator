from typing import Any

from django.db import transaction
from django.db.models import Q
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.http.response import HttpResponseBase
from django.utils.datastructures import MultiValueDictKeyError
from drf_spectacular.utils import extend_schema, OpenApiExample, inline_serializer, PolymorphicProxySerializer, \
    OpenApiResponse
from rest_framework import status
from rest_framework.exceptions import APIException, NotFound, ParseError, PermissionDenied, ValidationError
from rest_framework.response import Response
from rest_framework.fields import UUIDField, CharField

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
from ..serializers.model import ModelSerializer, load_and_create_model_request, GlobalModelSerializer, \
    MeanModelSerializer, SWAGModelSerializer, ModelSerializerNoWeights
from ..openapi import error_response_403


class Model(ViewSet):

    serializer_class = PolymorphicProxySerializer(
        "PolymorphicModelSerializer",
        [
            GlobalModelSerializer(many=True),
            MeanModelSerializer(many=True),
            SWAGModelSerializer(many=True)
        ],
        None,
        many=False
    )

    def get_models(self, request: HttpRequest) -> HttpResponse:
        """
        Get a list of all models which are related to the requested user.
        A model is related to an user if it is owned by the user or if the user
        is the actor or a participant of the models training.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: model list as json response
        """
        user_ids = Training.objects.filter(
            Q(actor=request.user) | Q(participants=request.user)
        ).distinct().values_list("model__id", flat=True)
        models = ModelDB.objects.filter(Q(owner=request.user) | Q(id__in=user_ids)).distinct()
        serializer = ModelSerializer(models, many=True)
        return Response(serializer.data)

    @extend_schema(
        responses={
            status.HTTP_200_OK: ModelSerializerNoWeights(),
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
            status.HTTP_403_FORBIDDEN: error_response_403,
        },
    )
    def get_metadata(self, _request: HttpRequest, id: str) -> HttpResponse:
        """
        Get model meta data.

        Args:
            request (HttpRequest):  request object
            id (str):  model uuid

        Returns:
            HttpResponse: model meta data as json response
        """
        model = get_entity(ModelDB, pk=id)
        serializer = ModelSerializer(model)
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
        Download the whole model as pickle file.

        Args:
            request (HttpRequest):  request object
            id (str):  model uuid

        Returns:
            HttpResponseBase: model as file response or 404 if model not found
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
        response["Content-Disposition"] = f'filename="model-{id}.pkl"'
        return response

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "model_file": {"type": "string", "format": "binary"}},
            },
        },
        responses={
            status.HTTP_200_OK: inline_serializer("ModelUploadSerializer", fields={
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

        The file should be a pickled torch.nn.Module.

        Args:
            request (HttpRequest):  request object

        Returns:
            HttpResponse: upload success message as json response
        """
        model = load_and_create_model_request(request)
        return JsonResponse({
            "detail": "Model Upload Accepted",
            "model_id": str(model.id),
        }, status=status.HTTP_201_CREATED)

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
    def _save_swag_stats(fst_moment: bytes, snd_moment: bytes,
                         model: GlobalModelDB, client: UserDB, sample_size: int):
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
        if client.id not in [p.id for p in train.participants.all()]:
            raise PermissionDenied(f"You are not a participant of training {train.id}!")
        if train.state != expected_state:
            raise ValidationError(f"Training with ID {train.id} is in state {train.state}")
        if int(round_num) != train.model.round:
            raise ValidationError(f"Training with ID {train.id} is not currently in round {round_num}")
