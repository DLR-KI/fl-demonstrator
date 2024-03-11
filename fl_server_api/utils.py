from django.db import models
from django.core.exceptions import ObjectDoesNotExist, ValidationError as ValidationException
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest
from collections.abc import Callable
import json
from rest_framework.exceptions import ParseError
from types import FunctionType, MethodType
from typing import Any, Optional, Type, TypeVar, Union


_TModel = TypeVar("_TModel", bound=models.Model)


def get_entity(
    cls: Type[_TModel],
    error_message: Optional[str] = None,
    error_identifier: Optional[str] = None,
    *args,
    **kwargs
) -> _TModel:
    """
    Get model instance.

    Try to get a model instance. Similar to `<Model>.objects.get()`.
    But, catch and evaluate all kinds of exceptions which `<Model>.objects.get()` throws,
    combine it and build an response including an error message inside a `HttpResponseBadRequest`
    and throw `EntityNotFoundException` (one single Exception).

    Args:
        cls (Type[_TModel]): model class where the entity should be found
        error_message (Optional[str], optional): message added to `HttpResponseBadRequest` if an error occurs.
            If `None` a "<Model> <error_identifier> not found." message will be created.
        error_identifier (Optional[str], optional): model identifier or field name for the error message creation
            if an error occurs.
            If `None` the arguments of `<Model>.objects.get()` are searched for "pk" or "id" as fallback.

    Raises:
        ValidationError: If `<Model>.objects.get()` raise any kind of error.

    Returns:
        _TModel: <Model> instance

    Examples:
        ```python
        def get_user(self, request: HttpRequest, id: int) -> HttpResponseBase:
            user = get_entity(User, pk=id)  # EntityNotFoundException will be handled by middleware
            serializer = UserSerializer(user)
            return Response(serializer.data)
        ```
    """
    try:
        return cls.objects.get(*args, **kwargs)
    except (ObjectDoesNotExist, ValidationException):
        if error_message:
            raise ParseError(error_message)

        if error_identifier is None:
            error_identifier = kwargs.get("pk", kwargs.get("id", None))
        msg = f"{cls.__name__} {error_identifier} not found."
        raise ParseError(msg)


def get_file(
    request: HttpRequest,
    name: str,
    validator: Optional[Callable[[HttpRequest, str, UploadedFile, bytes], Optional[str]]] = None,
    **kwargs
) -> bytes:
    """
    Try to get a single uploaded file and optional check it.

    Args:
        request (HttpRequest): http request where the file should be found
        name (str): name or key of the uploaded file
        validator (Optional[Callable[[HttpRequest, str, UploadedFile, bytes], Optional[str]]]], optional):
            file validation method, returns error message if file is not valid; otherwise `None`
        kwargs: additional validator arguments

    Raises:
        ParseError: If no file was found or validator return error messages.

    Returns:
        bytes: file content
    """
    uploaded_file = request.FILES.get(name)
    if not uploaded_file or not uploaded_file.file:
        raise ParseError(f"No uploaded file '{name}' found.")
    file_content = uploaded_file.file.read()
    if validator:
        error_message = validator(request, name, uploaded_file, file_content, **kwargs)
        if error_message:
            raise ParseError(error_message)
    return file_content


def is_json(s: Union[str, bytes, bytearray]) -> Any:
    """
    Check if the given argument s is a valid json.

    Args:
        s (Union[str, bytes, bytearray]): validation objective

    Returns:
        Any: Deserialize object of objective `s` is a valid json; otherwise `False`
    """
    try:
        return json.loads(s)
    except ValueError:
        return False


def fullname(cls_or_obj_or_fn: Union[type, object, Callable]) -> str:
    """
    Get full qualified name of class, object or function.

    Args:
        cls_or_obj_or_fn (Union[type, object, Callable]): Inspection target.

    Returns:
        str: qualified name
    """
    if isinstance(cls_or_obj_or_fn, (type, FunctionType, MethodType)):
        cls = cls_or_obj_or_fn  # class or function
    else:
        cls = cls_or_obj_or_fn.__class__  # object

    module = cls.__module__
    # if module == "builtins":  # or module == "__main__":  # NOSONAR: S125
    #     # avoid outputs like "builtins.str"
    #     return cls.__qualname__
    return module + "." + cls.__qualname__
