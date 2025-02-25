import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseModel(ABC):

    def __init__(self, model_name: str):
        pass

    def generate(self, prompt):
        pass

    def generate_p2p(self, prompt, image_path):
        pass

    def generate_s2p(self, prompt, item, input_image):
        pass