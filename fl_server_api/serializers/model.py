# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.http import HttpRequest
from logging import getLogger
from rest_framework import serializers
from rest_framework.exceptions import ValidationError
from torchinfo import summary
from torchinfo.model_statistics import ModelStatistics
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from fl_server_core.models import (
    GlobalModel,
    LocalModel,
    MeanModel,
    Model,
    SWAGModel,
)
from fl_server_core.models.model import TModel
from fl_server_core.utils.torch_serialization import to_torch_module

from ..utils import get_file


def _model_stats_to_dict(stats: ModelStatistics) -> Dict[str, Any]:
    """
    Convert model statistics to a dictionary.

    Args:
        stats (ModelStatistics): The model statistics.

    Returns:
        Dict[str, Any]: The dictionary representation of the model statistics.
    """
    return dict(
        input_size=stats.input_size,
        total_input=stats.total_input,
        total_mult_adds=stats.total_mult_adds,
        total_output_bytes=stats.total_output_bytes,
        total_param_bytes=stats.total_param_bytes,
        total_params=stats.total_params,
        trainable_params=stats.trainable_params,
        summary_list=[
            dict(
                # Metadata
                layer_id=layer.layer_id,
                class_name=layer.class_name,
                depth=layer.depth,
                depth_index=layer.depth_index,
                executed=layer.executed,
                var_name=layer.var_name,
                is_leaf_layer=layer.is_leaf_layer,
                contains_lazy_param=layer.contains_lazy_param,
                # Statistics
                is_recursive=layer.is_recursive,
                input_size=layer.input_size,
                output_size=layer.output_size,
                kernel_size=layer.kernel_size,
                trainable_params=layer.trainable_params,
                num_params=layer.num_params,
                param_bytes=layer.param_bytes,
                output_bytes=layer.output_bytes,
                macs=layer.macs,
            )
            for layer in stats.summary_list
        ],
    )


def _create_model_serializer(cls: Type[TModel], *, name: Optional[str] = None) -> Type[serializers.ModelSerializer]:
    """
    Create a generic Model serializer.

    Args:
        cls (Type[TModel]): Database Model model.
        name (Optional[str], optional): Serializer name. Defaults to `None`.

    Returns:
        Type[serializers.ModelSerializer]: Model model serializer.
    """
    class ModelSerializer(serializers.ModelSerializer):

        class Meta:
            model = cls
            exclude = ["polymorphic_ctype"]

        def to_representation(self, instance):
            # only return the whole model (weights) if requested
            data = serializers.ModelSerializer.to_representation(self, instance)
            if not self.context.get("with-weights", False):
                del data["weights"]
            if self.context.get("with-stats", False):
                data["stats"] = self.analyze_torch_model(instance)
            if isinstance(instance, GlobalModel):
                data["has_preprocessing"] = bool(instance.preprocessing)
            return data

        def analyze_torch_model(self, instance: Model):
            if not isinstance(instance, GlobalModel) or not instance.input_shape:
                return None
            # try to analyze the model stats
            try:
                torch_model = instance.get_torch_model()
                input_shape: List[int] = [1 if x is None else x for x in instance.input_shape]
                stats = summary(torch_model, input_size=input_shape, verbose=0)
                return _model_stats_to_dict(stats)
            except Exception as e:
                getLogger("fl.server").warning(f"Failed to analyze model {instance.id}: {e}")
            return None

    return type(name or f"{cls.__name__}Serializer", (ModelSerializer, ), {})


# Create generic serializers for database model models.
GlobalModelSerializer = _create_model_serializer(GlobalModel)
LocalModelSerializer = _create_model_serializer(LocalModel)
ModelFallbackSerializer = _create_model_serializer(Model, name="ModelFallbackSerializer")

MeanModelSerializer = _create_model_serializer(MeanModel)
SWAGModelSerializer = _create_model_serializer(SWAGModel)


class ModelSerializer(serializers.ModelSerializer):
    """
    A common serializer for the polymorphic Model class.

    The model classes `GlobalModel`, `LocalModel`, `MeanModel`, and `SWAGModel` are supported.
    """

    class Meta:
        model = Model
        exclude = ["polymorphic_ctype"]

    @classmethod
    def get_serializer(cls, model: Type[Model]):
        """
        Get the appropriate serializer based on the model type.

        Args:
            model (Type[Model]): The model class.

        Returns:
            Type[serializers.ModelSerializer]: The serializer class for the model.
        """
        if model == GlobalModel:
            return GlobalModelSerializer
        if model == LocalModel:
            return LocalModelSerializer
        if model == MeanModel:
            return MeanModelSerializer
        if model == SWAGModel:
            return SWAGModelSerializer
        getLogger("fl.server").warning(f"Using fallback model serializer for {model.__name__}")
        return ModelFallbackSerializer

    def to_representation(self, instance: Model):
        """
        Generate a dictionary representation of the corresponding model instance.

        Args:
            instance (Model): The Model instance.

        Returns:
            dict: The dictionary representation of the Model instance.
        """
        serializer = self.get_serializer(instance.__class__)
        return serializer(instance, context=self.context).data


class ModelSerializerNoWeights(ModelSerializer):
    """
    A model serializer that excludes the model weights.
    """

    class Meta:
        model = Model
        exclude = ["polymorphic_ctype", "weights"]
        include = ["has_preprocessing"]


class ModelSerializerNoWeightsWithStats(ModelSerializerNoWeights):
    """
    A model serializer that excludes the model weights but includes the model statistics.
    """

    class Meta:
        model = Model
        exclude = ["polymorphic_ctype", "weights"]
        include = ["stats"]
        include = ["has_preprocessing", "stats"]


#######################################################################################################################
# POST BODY SERIALIZERS #


def load_and_create_model_request(request: HttpRequest) -> Model:
    """
    Load and create a model from a Django request.

    Args:
        request (HttpRequest): The request object.

    Returns:
        Model: The created model.
    """
    model_cls, data = load_model_request(request)
    if model_cls is MeanModel:
        sub_models = data.pop("models")
        model = model_cls.objects.create(**data)
        model.models.set(sub_models)
        return model
    return model_cls.objects.create(**data)


def load_model_request(request: HttpRequest) -> Tuple[Type[Model], Dict[str, Any]]:
    """
    Load model data from a Django request.

    Args:
        request (HttpRequest): The request object.

    Returns:
        Tuple[Type[Model], Dict[str, Any]]: The model class and parsed request data.
    """
    if request.content_type == "application/json":
        data = request.data
    else:  # multipart/form-data
        data = request.POST.dict()
    data["owner"] = request.user
    model_type = data.pop("type", "GLOBAL").upper()

    # name and description are required
    if "name" not in data:
        raise ValidationError("Missing required field: name")
    if "description" not in data:
        raise ValidationError("Missing required field: description")

    # round is an optional field (it may not even make sense to be able to set it)
    if "round" not in data:
        data["round"] = 0

    # QueryDict removes all "None" entities.
    # Therefore, we need to manually parse the data to dict (the "normal" python way) and get the correct data
    if request.content_type == "application/json":
        data["input_shape"] = dict(request.data).get("input_shape", None)
    else:  # multipart/form-data
        data["input_shape"] = dict(request.POST).get("input_shape", None)
    # input_shape contains None as String but this is not parseable to a nullable int later -> parsing
    if "input_shape" in data:
        if data["input_shape"] is None:
            # nothing to do
            pass
        elif isinstance(data["input_shape"], str) and data["input_shape"].upper() in ["NONE", "NULL"]:
            data["input_shape"] = None
        elif not isinstance(data["input_shape"], Sequence):
            data["input_shape"] = None
        elif len(data["input_shape"]) < 1:
            data["input_shape"] = None
        else:
            data["input_shape"] = [None if str(x).upper() in ["NONE", "NULL"] else x for x in data["input_shape"]]

    # model_file is required except for MEAN models
    if model_type != "MEAN":
        data["weights"] = get_file(request, "model_file")
        verify_model_object(data["weights"])

    # model_preprocessing_file is optional
    data["preprocessing"] = None
    if request.FILES.__contains__("model_preprocessing_file"):
        data["preprocessing"] = get_file(request, "model_preprocessing_file")
        verify_model_object(data["preprocessing"], "preprocessing")

    # MEAN models require a list of model UUIDs
    if model_type == "MEAN":
        data = _parse_mean_model_models(data)

    # return class type and parsed data
    match model_type:
        case "SWAG": return SWAGModel, data
        case "MEAN": return MeanModel, data
        case _: return GlobalModel, data


def verify_model_object(model: bytes, file_type_name: str = "model") -> None:
    """
    Verify the model object.

    Args:
        model (bytes): The model object.
        file_type_name (str, optional): The file type name. Defaults to "model".

    Raises:
        ValidationError: If the model object is invalid.
    """
    try:
        to_torch_module(model)
    except Exception as e:
        raise ValidationError(f"Invalid {file_type_name} file: {e}") from e


def _parse_mean_model_models(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the models field of a mean model request.

    Args:
        data (Dict[str, Any]): The request data.

    Returns:
        Dict[str, Any]: The parsed request data.

    Raises:
        ValidationError: If the models field is missing or invalid.
    """
    if "models" not in data:
        raise ValidationError("Missing required field: models")
    models = data["models"]
    if not isinstance(models, list) or not all([isinstance(m, str) for m in models]):
        raise ValidationError("Invalid type for field: models")
    data["models"] = Model.objects.filter(id__in=models)
    if data["models"].count() != len(models):
        raise ValidationError("Invalid model UUIDs found.")
    if not all([m.owner.id == data["owner"].id for m in data["models"]]):
        raise ValidationError("Invalid model selection. Insufficient model permissions.")
    return data
