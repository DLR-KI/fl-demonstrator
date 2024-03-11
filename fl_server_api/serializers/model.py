from django.http import HttpRequest
from logging import getLogger
from rest_framework import serializers
from typing import Any, Dict, Optional, Tuple, Type

from fl_server_core.models import (
    GlobalModel,
    LocalModel,
    MeanModel,
    Model,
    SWAGModel,
)
from fl_server_core.models.model import TModel

from ..utils import get_file


def _create_model_serializer(cls: Type[TModel], *, name: Optional[str] = None) -> Type[serializers.ModelSerializer]:
    """
    Create a generic Model serializer.

    Args:
        cls (Type[TModel]): database Model model
        name (Optional[str], optional): serializer name. Defaults to `None`.

    Returns:
        Type[serializers.ModelSerializer]: Model model serializer
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
            return data

    return type(name or f"{cls.__name__}Serializer", (ModelSerializer, ), {})


GlobalModelSerializer = _create_model_serializer(GlobalModel)
LocalModelSerializer = _create_model_serializer(LocalModel)
ModelFallbackSerializer = _create_model_serializer(Model, name="ModelFallbackSerializer")

MeanModelSerializer = _create_model_serializer(MeanModel)
SWAGModelSerializer = _create_model_serializer(SWAGModel)


class ModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = Model
        exclude = ["polymorphic_ctype"]

    @classmethod
    def get_serializer(cls, model: Type[Model]):
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
        serializer = self.get_serializer(instance.__class__)
        return serializer(instance, context=self.context).data


class ModelSerializerNoWeights(ModelSerializer):
    class Meta:
        model = Model
        exclude = ["polymorphic_ctype", "weights"]


#######################################################################################################################
# POST BODY SERIALIZERS #


def load_and_create_model_request(request: HttpRequest) -> Model:
    """
    Load and create a model from a Django request.

    Args:
        request (HttpRequest): request object

    Returns:
        Model: created model
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
        request (HttpRequest): request object

    Returns:
        Tuple[Type[Model], Dict[str, Any]]: model class and parsed request data
    """
    if request.content_type == "application/json":
        data = request.data
    else:
        # multipart/form-data
        data = request.POST.dict()
    data["owner"] = request.user
    model_type = data.pop("type", "GLOBAL").upper()

    # name and description are required
    if "name" not in data:
        raise ValueError("Missing required field: name")
    if "description" not in data:
        raise ValueError("Missing required field: description")

    # round is an optional field (it may not even make sense to be able to set it)
    if "round" not in data:
        data["round"] = 0

    if "input_shape" not in data:
        data["input_shape"] = "[None]"

    # model_file is required except for MEAN models
    if model_type != "MEAN":
        data["weights"] = get_file(request, "model_file")

    # MEAN models require a list of model UUIDs
    if model_type == "MEAN":
        data = _parse_mean_model_models(data)

    # return class type and parsed data
    match model_type:
        case "SWAG": return SWAGModel, data
        case "MEAN": return MeanModel, data
        case _: return GlobalModel, data


def _parse_mean_model_models(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the models field of a mean model request.

    Args:
        data (Dict[str, Any]): request data

    Returns:
        Dict[str, Any]: parsed request data
    """
    if "models" not in data:
        raise ValueError("Missing required field: models")
    models = data["models"]
    if not isinstance(models, list) or not all([isinstance(m, str) for m in models]):
        raise ValueError("Invalid type for field: models")
    data["models"] = Model.objects.filter(id__in=models)
    if data["models"].count() != len(models):
        raise ValueError("Invalid model UUIDs found.")
    if not all([m.owner.id == data["owner"].id for m in data["models"]]):
        raise ValueError("Invalid model selection. Insufficient model permissions.")
    return data
