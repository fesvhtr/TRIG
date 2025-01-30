import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseModel(ABC):

    def __init__(self, **kwargs):
        pass

    def generate(self, prompt, **kwargs):
        pass