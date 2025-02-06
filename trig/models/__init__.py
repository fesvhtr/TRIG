from .base import BaseModel
import importlib

AVAILABLE_Model = {
    'base': 'base.BaseModel', 
    # t2i models 12000 pics
    'dalle3': 'dalle.DALLE3Model',
    'sdxl': 'diffuser_models.SDXLModel', # 30h - 1024px, A6000, 50step
    'omnigen': 'diffuser_models.OmniGenModel', # 200h - 1024px, A6000, 50step ???
    'pixart_sigma': 'diffuser_models.PixartSigmaModel', # 20h - 1024px, A6000, 20step
    'onediffusion': 'diffuser_models.OneDiffusionModel', # 150h - 1024px, A6000, 50step
    'sana': 'diffuser_models.SanaModel', # 20h - 1024px, A6000, 20step
    'sd35': 'diffuser_models.SD35Model', # 140h - 1024px, A6000, 28step
    'flux': 'diffuser_models.FLUXModel', # 300h - 1024px, A6000, 50step
    'janus': 'januspro.JanusModel', # 
    # p2p models
    # subjects models
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