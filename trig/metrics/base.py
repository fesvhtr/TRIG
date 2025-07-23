import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
import numpy as np


class BaseMetric(ABC):

    def __init__(self, dimension: Any = None, **kwargs):
        self.dimension = dimension
        super().__init__()

    @abstractmethod
    def compute(self, image_path: str, prompt: str, *args) -> float:
        pass

    @abstractmethod
    def compute_batch(self, task: str, promp_data: List[Dict[str, Any]]) -> Dict[str, float]:
        pass
