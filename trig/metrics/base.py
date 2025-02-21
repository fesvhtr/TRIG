import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
import numpy as np


class BaseMetric(ABC):

    def __init__(self, dimension, **kwargs):
        self.dimension = dimension
        super().__init__()

    @abstractmethod
    def compute(self, image_path: str, prompt: str) -> float:
        pass

    @abstractmethod
    def compute_batch(self, images: List[str], prompts: List[str], dimension: Any = None) -> np.ndarray:
        pass
