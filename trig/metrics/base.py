import abc
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
import numpy as np


class BaseMetric(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def compute(self, image_path: str, prompt: str) -> float:
        """
        计算单张图片的度量分数
        
        Args:
            image_path: 图片路径
            prompt: 提示文本
            
        Returns:
            float: 度量分数
        """
        pass

    @abstractmethod
    def compute_batch(self, images: List[str], prompts: List[str], dimension: Any = None) -> np.ndarray:
        """
        批量计算图片的度量分数
        
        Args:
            images: 图片路径列表
            prompts: 提示文本列表
            dimension: 可选的维度信息
            
        Returns:
            np.ndarray: 度量分数数组
        """
        pass
