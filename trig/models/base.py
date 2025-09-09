import abc
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseModel(ABC):

    def __init__(self, model_name: str = None):
        pass
    
    def get_model_path(self, default_model_id: str, config_key: str) -> str:
        # 优先检查配置文件中的路径
        if hasattr(self, '_local_config') and config_key in self._local_config:
            local_path = self._local_config[config_key]
            if os.path.exists(local_path):
                print(f"Using local model path from config: {local_path}")
                return local_path
            else:
                print(f"Warning: Local path {local_path} from config does not exist, falling back to HuggingFace")
        
        # 如果没有配置文件或配置中没有该路径，使用HuggingFace模型ID
        print(f"Using HuggingFace model: {default_model_id}")
        return default_model_id
    
    def load_local_config(self, config_path: str = None):
        if not hasattr(self, '_local_config'):
            self._local_config = {}
        
        # 默认配置文件路径
        if config_path is None:
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            config_path = os.path.join(project_root, 'config', 'local_models.env')
        
        if os.path.exists(config_path):
            print(f"Loading local model config from: {config_path}")
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self._local_config[key.strip()] = value.strip()
        else:
            print(f"No local config found at: {config_path}")
        
        return self._local_config

    def generate(self, prompt):
        pass

    def generate_p2p(self, prompt, image_path):
        pass

    def generate_s2p(self, prompt, item, input_image):
        pass