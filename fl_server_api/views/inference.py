from django.http import HttpRequest, HttpResponse
from drf_spectacular.utils import inline_serializer, extend_schema, OpenApiExample
import json
import pickle
from rest_framework import status
from rest_framework.exceptions import APIException, UnsupportedMediaType, ValidationError
from rest_framework.fields import ListField, DictField, FloatField, CharField
import torch
from typing import Any, Dict, Tuple, Type

from fl_server_core.exceptions import TorchDeserializationException
from fl_server_core.models import Model
from fl_server_ai.uncertainty import get_uncertainty_class, UncertaintyBase
from ..serializers.generic import ErrorSerializer

from .base import ViewSet
from ..utils import get_entity


class Inference(ViewSet):

    serializer_class = inline_serializer("InferenceSerializer", fields={
        "inference": ListField(child=ListField(child=FloatField())),
        "uncertainty": DictField(child=FloatField())
    })

    @extend_schema(
        request=inline_serializer(
            "InferenceJsonSerializer",
            fields={
                "model_id": CharField(),
                "model_input": ListField(child=ListField(child=FloatField()))
            }
        ),
        responses={
            status.HTTP_200_OK: serializer_class,
            status.HTTP_400_BAD_REQUEST: ErrorSerializer,
        },
        examples=[
            OpenApiExample("JSON Example", value={
                "model_id": "mymodel",
                "model_input": [
                    [1.0, 2.3, -0.4, 3],
                    [0.01, 9.7, 5.6, 7]
                ]
            }, request_only=True),
        ]
    )
    def inference(self, request: HttpRequest) -> HttpResponse:
        """
        Processes a request to do inference on a existing model.

        Note that this endpoint can process both JSON data as well as formdata with an
        attached pickled input tensor.
        Further note that for the provided example to run, the model "mymodel" must have been created
        via the model-endpoint first, and the user must be authorized to access it.

        Args:
            request (HttpRequest): request object

        Returns:
            HttpResponse: (pickled) results of the inference
        """
        match request.content_type.lower():
            case s if s.startswith("multipart/form-data"):
                return self._process_post(request)
            case s if s.startswith("application/json"):
                return self._process_post_json(request)
            case _:
                # If the content type is specified, but not supported, return 415
                self._logger.error(f"Unknown Content-Type '{request.content_type}'")
                raise UnsupportedMediaType(
                    "Only Content-Type 'application/json' and 'multipart/form-data' is supported."
                )

    def _process_post(self, request: HttpRequest) -> HttpResponse:
        try:
            model_id = request.POST["model_id"]
            uploaded_file = request.FILES.get("model_input")
            if not uploaded_file or not uploaded_file.file:
                raise ValidationError("No uploaded file 'model_input' not found.")
            feature_vectors = uploaded_file.file.read()
        except Exception as e:
            self._logger.error(e)
            raise ValidationError("Inference Request could not be interpreted!")

        model = get_entity(Model, pk=model_id)
        input_tensor = pickle.loads(feature_vectors)
        _, inference, uncertainty = self.do_inference(model, input_tensor)
        response_bytes = pickle.dumps(dict(inference=inference, uncertainty=uncertainty))
        return HttpResponse(response_bytes, content_type="application/octet-stream")

    def _process_post_json(self, request: HttpRequest, body: Any = None) -> HttpResponse:
        try:
            body = body or json.loads(request.body)
            model_id = body["model_id"]
            model_input = body["model_input"]
        except Exception as e:
            self._logger.error(e)
            raise ValidationError("Inference Request could not be interpreted!")

        model = get_entity(Model, pk=model_id)
        input_tensor = torch.as_tensor(model_input)
        uncertainty_cls, inference, uncertainty = self.do_inference(model, input_tensor)
        return HttpResponse(uncertainty_cls.to_json(inference, uncertainty), content_type="application/json")

    def do_inference(
        self, model: Model, input_tensor: torch.Tensor
    ) -> Tuple[Type[UncertaintyBase], torch.Tensor, Dict[str, Any]]:
        try:
            uncertainty_cls = get_uncertainty_class(model)
            inference, uncertainty = uncertainty_cls.prediction(input_tensor, model)
            return uncertainty_cls, inference, uncertainty
        except TorchDeserializationException as e:
            raise APIException(e)
        except Exception as e:
            self._logger.error(e)
            raise APIException("Internal Server Error occurred during inference!")
