import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseModel(ABC):

    def __init__(self, model_name: str):
        pass

    def generate(self, prompt):
        pass

    def generate_p2p(self, image_path, prompt):
        pass

    def generate_sub(self, image_path, prompt):
        pass