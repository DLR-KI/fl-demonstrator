# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from django.http import HttpRequest, HttpResponse
from drf_spectacular.utils import inline_serializer, extend_schema, OpenApiExample
import json
from io import BytesIO
import pickle
from PIL import Image
from rest_framework import status
from rest_framework.exceptions import APIException, UnsupportedMediaType, ValidationError
from rest_framework.fields import CharField, ChoiceField, DictField, FloatField, ListField
import torch
from torchvision.transforms.functional import to_tensor
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from fl_server_ai.uncertainty import get_uncertainty_class, UncertaintyBase
from fl_server_core.exceptions import TorchDeserializationException
from fl_server_core.models import Model, GlobalModel, LocalModel
from fl_server_core.utils.logging import disable_logger
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
                "return_format": ChoiceField(["binary", "json"])
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
        Performs inference on the provided model and input data.

        This method takes in an HTTP request containing the necessary metadata and input data,
        performs any required preprocessing on the input data, runs the inference using the specified model,
        and returns a response in the format specified by the `return_format` parameter including
        possible uncertainty measurements if defined.

        Args:
            request (HttpRequest): The current HTTP request.

        Returns:
            HttpResponse: A HttpResponse containing the result of the inference as well as its uncertainty.
        """
        request_body, is_json = self._get_handle_content_type(request)
        model, preprocessing, input_shape, return_format = self._get_inference_metadata(
            request_body,
            "json" if is_json else "binary"
        )
        model_input = self._get_model_input(request, request_body)

        if preprocessing:
            model_input = preprocessing(model_input)
        else:
            # if no preprocessing is defined, at least try to convert/interpret the model_input as
            # PyTorch tensor, before raising an exception
            model_input = self._try_cast_model_input_to_tensor(model_input)
        self._validate_model_input_after_preprocessing(model_input, input_shape, bool(preprocessing))

        uncertainty_cls, inference, uncertainty = self._do_inference(model, model_input)
        return self._make_response(uncertainty_cls, inference, uncertainty, return_format)

    def _get_handle_content_type(self, request: HttpRequest) -> Tuple[dict, bool]:
        """
        Handles HTTP request body based on their content type.

        This function checks if the request content type is either `application/json`
        or `multipart/form-data`. If it matches, it returns the corresponding data and
        a boolean indicating whether it's JSON (True) or multipart/form-data (False).

        Args:
            request (HttpRequest): The request.

        Returns:
            tuple: A tuple containing the parsed data and a boolean indicating the content type.
                * If content type is `application/json`, returns the JSON payload as a Python object (dict)
                and True to indicate it's JSON.
                * If content type is `multipart/form-data`, returns the request POST data and False.

        Raises:
            UnsupportedMediaType: If an unknown content type is specified, raising an error with
                details on supported types (`application/json` and `multipart/form-data`).
        """
        match request.content_type.lower():
            case s if s.startswith("multipart/form-data"):
                return request.POST, False
            case s if s.startswith("application/json"):
                return json.loads(request.body), True

        # if the content type is specified, but not supported, return 415
        self._logger.error(f"Unknown Content-Type '{request.content_type}'")
        raise UnsupportedMediaType(
            "Only Content-Type 'application/json' and 'multipart/form-data' is supported."
        )

    def _get_inference_metadata(
        self,
        request_body: dict,
        return_format_default: Literal["binary", "json"]
    ) -> Tuple[Model, Optional[torch.nn.Module], Optional[List[Optional[int]]], str]:
        """
        Retrieves inference metadata based on the content of the provided request body.

        This method checks if a `model_id` is present in the request body and retrieves
        the corresponding model entity. It then determines the return format based on the
        request body or default to one of the two supported formats (`binary` or `json`).

        Args:
            request_body (dict): The data sent with the request, containing at least `model_id`.
            return_format_default (Literal["binary", "json"]): The default return format to use if not specified in
                the request body.

        Returns:
            Tuple[Model, Optional[torch.nn.Module], Optional[List[Optional[int]]], str]: A tuple containing:
                * The retrieved model entity.
                * The global model's preprocessing torch module (if applicable).
                * The input shape of the global model (if applicable).
                * The return format (`binary` or `json`).

        Raises:
            ValidationError: If no valid `model_id` is provided in the request body, or if an unknown return format
                is specified.
        """
        if "model_id" not in request_body:
            self._logger.error("No 'model_id' provided in request.")
            raise ValidationError("No 'model_id' provided in request.")
        model_id = request_body["model_id"]
        model = get_entity(Model, pk=model_id)

        return_format = request_body.get("return_format", return_format_default)
        if return_format not in ["binary", "json"]:
            self._logger.error(f"Unknown return format '{return_format}'. Supported are binary and json.")
            raise ValidationError(f"Unknown return format '{return_format}'. Supported are binary and json.")

        global_model: Optional[GlobalModel] = None
        if isinstance(model, GlobalModel):
            global_model = model
        elif isinstance(model, LocalModel):
            global_model = model.base_model
        else:
            self._logger.error("Unknown model type. Not a GlobalModel and not a LocalModel. Skip preprocessing.")

        preprocessing: Optional[torch.nn.Module] = None
        input_shape: Optional[List[Optional[int]]] = None
        if global_model:
            if global_model.preprocessing is not None:
                preprocessing = global_model.get_preprocessing_torch_model()
            if global_model.input_shape is not None:
                input_shape = global_model.input_shape

        return model, preprocessing, input_shape, return_format

    def _get_model_input(self, request: HttpRequest, request_body: dict) -> Any:
        """
        Retrieves and decodes the model input from either an uploaded file or the request body.

        Args:
            request (HttpRequest): The current HTTP request.
            request_body (dict): The parsed request body as a dictionary.

        Returns:
            Any: The decoded model input data.

        Raises:
            ValidationError: If no `model_input` is found in the uploaded file or the request body.
        """
        uploaded_file = request.FILES.get("model_input", None)
        if uploaded_file and uploaded_file.file:
            model_input = uploaded_file.file.read()
        else:
            model_input = request_body.get("model_input", None)
        if not model_input:
            raise ValidationError("No uploaded file 'model_input' found.")
        return self._try_decode_model_input(model_input)

    def _try_decode_model_input(self, model_input: Any) -> Any:
        """
        Attempts to decode the input `model_input` from various formats and returns it in a usable form.

        This function first tries to deserialize the input as a PyTorch tensor. If that fails, it attempts to
        decode the input as a base64-encoded string. If neither attempt is successful, the original input is returned.

        Args:
            model_input (Any): The input to be decoded, which can be in any format.

        Returns:
            Any: The decoded input, which may still be in an unknown format if decoding attempts fail.
        """
        # 1. try to deserialize model_input as PyTorch tensor
        try:
            with disable_logger(self._logger):
                model_input = to_torch_tensor(model_input)
        except Exception:
            pass
        # 2. try to decode model_input as base64
        try:
            is_base64, tmp_model_input = self._is_base64(model_input)
            if is_base64:
                model_input = tmp_model_input
        except Exception:
            pass
        # result
        return model_input

    def _try_cast_model_input_to_tensor(self, model_input: Any) -> Any:
        """
        Attempt to cast the given model input to a PyTorch tensor.

        This function tries to interpret the input in several formats:

        1. PIL Image (and later convert it to a PyTorch tensor, see 3.)
        2. PyTorch tensor via `torch.as_tensor`
        3. PyTorch tensor via torchvision `ToTensor` (supports e.g. PIL images)

        If none of these attempts are successful, the original input is returned.

        Args:
            model_input: The input data to be cast to a PyTorch tensor.
                Can be any type that can be converted to a tensor.

        Returns:
            A PyTorch tensor representation of the input data, or the original
            input if it cannot be converted.
        """
        def _try_to_pil_image(model_input: Any) -> Any:
            stream = BytesIO(model_input)
            return Image.open(stream)

        if isinstance(model_input, torch.Tensor):
            return model_input

        # In the following order, try to:
        # 1. interpret model_input as PIL image (and later to PyTorch tensor, see step 3),
        # 2. interpret model_input as PyTorch tensor,
        # 3. interpret model_input as PyTorch tensor via torchvision ToTensor (supports e.g. PIL images).
        for fn in [_try_to_pil_image, torch.as_tensor, to_tensor]:
            try:
                model_input = fn(model_input)  # type: ignore
            except Exception:
                pass
        return model_input

    def _is_base64(self, sb: str | bytes) -> Tuple[bool, bytes]:
        """
        Check if a string or bytes object is a valid Base64 encoded string.

        This function checks if the input can be decoded and re-encoded without any changes.
        If decoding and encoding returns the same result as the original input, it's likely
        that the input was indeed a valid Base64 encoded string.

        Note: This code is based on the reference implementation from the linked Stack Overflow answer.

        Args:
            sb (str | bytes): The input string or bytes object to check.

        Returns:
            Tuple[bool, bytes]: A tuple containing a boolean indicating whether the input is
                a valid Base64 encoded string and the decoded bytes if it is.

        References:
            https://stackoverflow.com/a/45928164
        """
        try:
            if isinstance(sb, str):
                # If there's any unicode here, an exception will be thrown and the function will return false
                sb_bytes = bytes(sb, "ascii")
            elif isinstance(sb, bytes):
                sb_bytes = sb
            else:
                raise ValueError("Argument must be string or bytes")
            decoded = base64.b64decode(sb_bytes)
            return base64.b64encode(decoded) == sb_bytes, decoded
        except Exception:
            return False, b""

    def _validate_model_input_after_preprocessing(
        self,
        model_input: Any,
        model_input_shape: Optional[List[Optional[int]]],
        preprocessing: bool
    ) -> None:
        """
        Validates the model input after preprocessing.

        Ensures that the provided `model_input` is a valid PyTorch tensor and its shape matches
        the expected`model_input_shape`.

        Args:
            model_input (Any): The model input to be validated.
            model_input_shape (Optional[List[Optional[int]]]): The expected shape of the model input.
                Can contain None values if not all dimensions are fixed (e.g. first dimension as batch size).
            preprocessing (bool): Whether a preprocessing model was defined or not. (Only for a better error message.)

        Raises:
            ValidationError: If the `model_input` is not a valid PyTorch tensor or
                its shape does not match the expected `model_input_shape`.
        """
        if not isinstance(model_input, torch.Tensor):
            msg = "Model input could not be casted or interpreted as a PyTorch tensor object"
            if preprocessing:
                msg += " and is still not a PyTorch tensor after preprecessing."
            else:
                msg += " and no preprecessing is defined."
            raise ValidationError(msg)

        if model_input_shape and not all(
            dim_input == dim_model
            for (dim_input, dim_model) in zip(model_input.shape, model_input_shape)
            if dim_model is not None
        ):
            raise ValidationError("Input shape does not match model input shape.")

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

        If return_type is "binary", a binary-encoded response will be generated using pickle.
        Otherwise, a JSON response will be generated by serializing the uncertainty object using its to_json method.

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

    def _do_inference(
        self, model: Model, input_tensor: torch.Tensor
    ) -> Tuple[Type[UncertaintyBase], torch.Tensor, Dict[str, Any]]:
        """
        Perform inference on a given input tensor using the provided model.

        This methods retrieves the uncertainty class, performs the prediction.
        The output of this method consists of:

        * The uncertainty class used for inference
        * The result of the model's prediction on the input tensor
        * Any associated uncertainty for the prediction

        Args:
            model (Model): The model to perform inference with.
            input_tensor (torch.Tensor): Input tensor to pass through the model.

        Returns:
            Tuple[Type[UncertaintyBase], torch.Tensor, Dict[str, Any]]:
                A tuple containing the uncertainty class, prediction result, and any associated uncertainty.

        Raises:
            APIException: If an error occurs during inference
        """
        try:
            uncertainty_cls = get_uncertainty_class(model)
            inference, uncertainty = uncertainty_cls.prediction(input_tensor, model)
            return uncertainty_cls, inference, uncertainty
        except TorchDeserializationException as e:
            raise APIException(e) from e
        except Exception as e:
            self._logger.error(e)
            raise APIException("Internal Server Error occurred during inference!") from e
