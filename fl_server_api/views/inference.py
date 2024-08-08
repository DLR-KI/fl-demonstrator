# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.http import HttpRequest, HttpResponse
from drf_spectacular.utils import inline_serializer, extend_schema, OpenApiExample
import json
import pickle
from rest_framework import status
from rest_framework.exceptions import APIException, UnsupportedMediaType, ValidationError
from rest_framework.fields import ListField, DictField, FloatField, CharField
import torch
from typing import Any, Dict, Tuple, Type

from fl_server_ai.uncertainty import get_uncertainty_class, UncertaintyBase
from fl_server_core.exceptions import TorchDeserializationException
from fl_server_core.models import Model, GlobalModel
from fl_server_core.utils.torch_serialization import to_torch_tensor

from .base import ViewSet
from ..serializers.generic import ErrorSerializer
from ..utils import get_entity


class Inference(ViewSet):
    """
    Inference ViewSet for performing inference on a model.
    """

    serializer_class = inline_serializer("InferenceSerializer", fields={
        "inference": ListField(child=ListField(child=FloatField())),
        "uncertainty": DictField(child=FloatField())
    })
    """The serializer for the ViewSet."""

    @extend_schema(
        request=inline_serializer(
            "InferenceJsonSerializer",
            fields={
                "model_id": CharField(),
                "model_input": ListField(child=ListField(child=FloatField())),
                "return_format": CharField()
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
                ],
                "return_format": "json"
            }, request_only=True),
        ]
    )
    def inference(self, request: HttpRequest) -> HttpResponse:
        """
        Processes a request to do inference on a existing model.

        This method checks the content type of the request and calls the appropriate method to process the request.
        If the content type is not supported, it raises an UnsupportedMediaType exception.

        This method can process both JSON data as well as formdata with an attached PyTorch serialised input tensor.
        For the provided example to run, the model "mymodel" must have been created via the model-endpoint first,
        and the user must be authorized to access it.

        Args:
            request (HttpRequest): The request.

        Returns:
            HttpResponse: The results of the inference.
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
        """
        Processes a POST request with form-data.

        Args:
            request (HttpRequest): The request.

        Returns:
            HttpResponse: The results of the inference.
        """
        try:
            model_id = request.POST["model_id"]
            uploaded_file = request.FILES.get("model_input")
            return_format = request.POST.get("return_format", "binary")
            assert return_format in ["binary", "json"]
            if not uploaded_file or not uploaded_file.file:
                raise ValidationError("No uploaded file 'model_input' not found.")
            feature_vectors = uploaded_file.file.read()
        except Exception as e:
            self._logger.error(e)
            raise ValidationError("Inference Request could not be interpreted!")

        # (Benedikt) imo, it does not make sense to allow inference on the local model updates, therefore I changed this
        model = get_entity(GlobalModel, pk=model_id)
        input_tensor = to_torch_tensor(feature_vectors)
        if model.preprocessing is not None:
            preprocessing = model.get_preprocessing_torch_model()
            input_tensor = preprocessing(input_tensor)

        if model.input_shape is not None:
            if not all(dim_input == dim_model for (dim_input, dim_model) in
                       zip(input_tensor.shape, model.input_shape) if dim_model is not None):
                raise ValidationError("Input shape does not match model input shape.")

        uncertainty_cls, inference, uncertainty = self.do_inference(model, input_tensor)
        return self._make_response(uncertainty_cls, inference, uncertainty, return_format)

    def _process_post_json(self, request: HttpRequest, body: Any = None) -> HttpResponse:
        """
        Processes a POST request with JSON data.

        Args:
            request (HttpRequest): The request.
            body (Any, optional): The request body. Defaults to None.

        Returns:
            HttpResponse: The results of the inference.
        """
        try:
            body = body or json.loads(request.body)
            return_format = body.get("return_format", "json")
            model_id = body["model_id"]
            model_input = body["model_input"]
        except Exception as e:
            self._logger.error(e)
            raise ValidationError("Inference Request could not be interpreted!")

        model = get_entity(GlobalModel, pk=model_id)
        input_tensor = torch.as_tensor(model_input)
        if model.preprocessing is not None:
            preprocessing = model.get_preprocessing_torch_model()
            input_tensor = preprocessing(input_tensor)

        if model.input_shape is not None:
            if not all(dim_input == dim_model for (dim_input, dim_model) in
                       zip(input_tensor.shape, model.input_shape) if dim_model is not None):
                raise ValidationError("Input shape does not match model input shape.")

        uncertainty_cls, inference, uncertainty = self.do_inference(model, input_tensor)
        return self._make_response(uncertainty_cls, inference, uncertainty, return_format)

    def _make_response(
        self,
        uncertainty_cls: Type[UncertaintyBase],
        inference: torch.Tensor,
        uncertainty: Any,
        return_type: str
    ) -> HttpResponse:
        """
        Build the response object with the result data.

        This method checks the return type and makes a response with the appropriate content type.

        Args:
            uncertainty_cls (Type[UncertaintyBase]): The uncertainty class.
            inference (torch.Tensor): The inference.
            uncertainty (Any): The uncertainty.
            return_type (str): The return type.

        Returns:
            HttpResponse: The inference result response.
        """
        if return_type == "binary":
            response_bytes = pickle.dumps(dict(inference=inference, uncertainty=uncertainty))
            return HttpResponse(response_bytes, content_type="application/octet-stream")

        return HttpResponse(uncertainty_cls.to_json(inference, uncertainty), content_type="application/json")

    def do_inference(
        self, model: Model, input_tensor: torch.Tensor
    ) -> Tuple[Type[UncertaintyBase], torch.Tensor, Dict[str, Any]]:
        """
        Performs inference on a model.

        This method gets the uncertainty class, performs prediction, and returns the uncertainty class, the inference,
        and the uncertainty.

        Args:
            model (Model): The model.
            input_tensor (torch.Tensor): The input tensor.

        Returns:
            Tuple[Type[UncertaintyBase], torch.Tensor, Dict[str, Any]]:
                The uncertainty class, the inference, and the uncertainty.
        """
        try:
            uncertainty_cls = get_uncertainty_class(model)
            inference, uncertainty = uncertainty_cls.prediction(input_tensor, model)
            return uncertainty_cls, inference, uncertainty
        except TorchDeserializationException as e:
            raise APIException(e)
        except Exception as e:
            self._logger.error(e)
            raise APIException("Internal Server Error occurred during inference!")
