from logging import getLogger
import pickle
from torch import Tensor
from torch.nn import Module
from typing import Any, Tuple, Type, TypeVar

from ..exceptions import TorchDeserializationException


T = TypeVar("T")


def to_torch(obj: Any, supported_types: Type[T] | Tuple[Type[T], ...]):
    t_obj = pickle.loads(obj)
    if isinstance(t_obj, supported_types):
        return t_obj
    getLogger("fl.server").error("Unpickled object is not a torch object.")
    raise TorchDeserializationException("Unpickled object is not a torch object.")


def from_torch(obj: Any, *args, **kwargs) -> bytes:
    return pickle.dumps(obj, *args, **kwargs)


def to_torch_module_or_tensor(obj: Any) -> Module | Tensor:
    return to_torch(obj, (Module, Tensor))


def from_torch_module_or_tensor(obj: Module | Tensor, *args, **kwargs) -> bytes:
    return from_torch(obj, *args, **kwargs)


def to_torch_module(obj: Any) -> Module:
    return to_torch(obj, Module)


def from_torch_module(obj: Module, *args, **kwargs) -> bytes:
    return from_torch(obj, *args, **kwargs)


def to_torch_tensor(obj: Any) -> Tensor:
    return to_torch(obj, Tensor)


def from_torch_tensor(obj: Tensor, *args, **kwargs) -> bytes:
    return from_torch(obj, *args, **kwargs)
