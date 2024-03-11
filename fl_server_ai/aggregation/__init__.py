from .base import Aggregation
from .mean import MeanAggregation
from .method import get_aggregation_class


__all__ = ["Aggregation", "get_aggregation_class", "MeanAggregation"]
