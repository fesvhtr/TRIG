import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseMetric(ABC):

    # FIXME add dimension
    def __init__(self, dimension=None, **kwargs):
        self.dimension = dimension
        super().__init__(**kwargs)

    @abstractmethod
    def compute(self, image_path, prompt):
        pass

    @abstractmethod
    def compute_batch(self, image_path_list, prompt_list):
        pass
