import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseModel(ABC):

    def __init__(self, model_name: str):
        pass

    @abstractmethod
    def generate(self, prompt):
        pass
