# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from logging import getLogger
import torch
from typing import Any, Tuple, Type, TypeVar
from typing_extensions import Buffer  # type: ignore[attr-defined]
# mypy issue see: https://github.com/python/mypy/issues/7182
import warnings

from ..exceptions import TorchDeserializationException


T = TypeVar("T")


def to_torch(obj: Any, supported_types: Type[T] | Tuple[Type[T], ...]):
    """
    Convert a serialized PyTorch object back into a PyTorch object.

    Args:
        obj (Any): The serialized PyTorch object.
        supported_types (Type[T] | Tuple[Type[T], ...]): The expected type or types of the PyTorch object.

    Returns:
        T: The deserialized PyTorch object.

    Raises:
        TorchDeserializationException: If there is an error during deserialization or if the deserialized
            object is not of the expected type.
    """
    obj = BytesIO(obj) if isinstance(obj, Buffer) else obj
    try:
        # torch.load support torch.nn.Module as well as torchscript (but with user warning)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'torch.load' received a zip file that looks like a TorchScript archive",
                category=UserWarning
            )
            t_obj = torch.load(obj)
    except Exception as e:
        getLogger("fl.server").error(f"Error loading torch object: {e}")
        raise TorchDeserializationException("Error loading torch object") from e
    if isinstance(t_obj, supported_types):
        return t_obj
    getLogger("fl.server").error("Loaded torch object is not of expected type.")
    raise TorchDeserializationException("Loaded torch object is not of expected type.")


def from_torch(obj: Any, *args, **kwargs) -> bytes:
    """
    Serialize a PyTorch object into bytes.

    Args:
        obj (Any): The PyTorch object to serialize.
        *args: Additional arguments to pass to the `torch.save` or `torch.jit.save` function.
        **kwargs: Additional keyword arguments to pass to the `torch.save` or `torch.jit.save` function.

    Returns:
        bytes: The serialized PyTorch object as bytes.
    """
    buffer = BytesIO()
    if is_torchscript_instance(obj):
        torch.jit.save(obj, buffer, *args, **kwargs)
    else:
        torch.save(obj, buffer, *args, **kwargs)
    return buffer.getvalue()


def to_torch_module_or_tensor(obj: Any) -> torch.nn.Module | torch.Tensor:
    """
    Convert a serialized PyTorch module or tensor back into a PyTorch module or tensor.

    Args:
        obj (Any): The serialized PyTorch module or tensor.

    Returns:
        torch.nn.Module | torch.Tensor: The deserialized PyTorch module or tensor.

    Raises:
        TorchDeserializationException: If there is an error during deserialization or if the deserialized
            object is not a PyTorch module or tensor.
    """
    return to_torch(obj, (torch.nn.Module, torch.Tensor))


def from_torch_module_or_tensor(obj: torch.nn.Module | torch.Tensor, *args, **kwargs) -> bytes:
    """
    Serialize a PyTorch module or tensor into bytes.

    Args:
        obj (torch.nn.Module | torch.Tensor): The PyTorch module or tensor to serialize.
        *args: Additional arguments to pass to the `torch.save` or `torch.jit.save` function.
        **kwargs: Additional keyword arguments to pass to the `torch.save` or `torch.jit.save` function.

    Returns:
        bytes: The serialized PyTorch module or tensor as bytes.
    """
    return from_torch(obj, *args, **kwargs)


def to_torch_module(obj: Any) -> torch.nn.Module:
    """
    Convert a serialized PyTorch module back into a PyTorch module.

    Args:
        obj (Any): The serialized PyTorch module.

    Returns:
        torch.nn.Module: The deserialized PyTorch module.

    Raises:
        TorchDeserializationException: If there is an error during deserialization or if the deserialized
            object is not a PyTorch module.
    """
    return to_torch(obj, torch.nn.Module)


def from_torch_module(obj: torch.nn.Module, *args, **kwargs) -> bytes:
    """
    Serialize a PyTorch module into bytes.

    Args:
        obj (torch.nn.Module): The PyTorch module to serialize.
        *args: Additional arguments to pass to the `torch.save` or `torch.jit.save` function.
        **kwargs: Additional keyword arguments to pass to the `torch.save` or `torch.jit.save` function.

    Returns:
        bytes: The serialized PyTorch module as bytes.
    """
    return from_torch(obj, *args, **kwargs)


def to_torch_tensor(obj: Any) -> torch.Tensor:
    """
    Convert a serialized PyTorch tensor back into a PyTorch tensor.

    Args:
        obj (Any): The serialized PyTorch tensor.

    Returns:
        torch.Tensor: The deserialized PyTorch tensor.

    Raises:
        TorchDeserializationException: If there is an error during deserialization or if the deserialized
            object is not a PyTorch tensor.
    """
    return to_torch(obj, torch.Tensor)


def from_torch_tensor(obj: torch.Tensor, *args, **kwargs) -> bytes:
    """
    Serialize a PyTorch tensor into bytes.

    Args:
        obj (torch.Tensor): The PyTorch tensor to serialize.
        *args: Additional arguments to pass to the `torch.save` or `torch.jit.save` function.
        **kwargs: Additional keyword arguments to pass to the `torch.save` or `torch.jit.save` function.

    Returns:
        bytes: The serialized PyTorch tensor as bytes.
    """
    return from_torch(obj, *args, **kwargs)


def is_torchscript_instance(obj: Any) -> bool:
    """
    Check if an object is an instance of `torch.jit.ScriptModule` or `torch.jit.ScriptFunction`.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: `True` if the object is an instance of `torch.jit.ScriptModule` or `torch.jit.ScriptFunction`,
            otherwise `False`.
    """
    return isinstance(obj, torch.jit.ScriptModule | torch.jit.ScriptFunction)
