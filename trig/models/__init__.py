from .base import BaseModel
import importlib

AVAILABLE_Model = {
    'base': 'base.BaseModel',
    'dalle3': 'dalle.DALLE3Model',
    'sdxl': 'diffuser_models.SDXLModel',
}

def import_model(model_name):
    """
    Dynamically import a specific class based on its name.

    :param model_name: The key in AVAILABLE_Model
    :return: The imported class.
    """
    if model_name not in AVAILABLE_Model:
        raise ValueError(f"Metric {model_name} not found in AVAILABLE_Metric.")

    try:
        full_path = f"trig.models.{AVAILABLE_Model[model_name]}"
        module_path, class_name = full_path.rsplit('.', 1)

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to import model {model_name}. Error: {e}")