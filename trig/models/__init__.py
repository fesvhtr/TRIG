from .base import BaseModel
import importlib

AVAILABLE_Model = {
    'base': 'base.BaseModel', 
    # general models
    'onediffusion': 'general_models.OneDiffusionModel', 
    'omnigen': 'general_models.OmniGenModel', 
    # t2i models 12300 pics
    'dalle3': 'text_to_image_models.DALLE3Model',
    'sdxl': 'text_to_image_models.SDXLModel',
    'pixart_sigma': 'text_to_image_models.PixartSigmaModel', 
    'sana': 'text_to_image_models.SanaModel', 
    'sd35': 'text_to_image_models.SD35Model', 
    'flux': 'text_to_image_models.FLUXModel', 
    'janus': 'text_to_image_modelso.JanusModel',
    'janus_flow': 'text_to_image_models.JanusFlowModel',
    # p2p models
    'instructp2p': 'image_editing_models.InstructPix2PixModel',
    'freediff': 'image_editing_models.FreeDiff',
    # subjects models
    'blipdiffusion': 'subject_driven_models.BlipDiffusionModel',
    'ssrencoder': 'subject_driven_models.SSREncoder',
    'omnicontrol': 'subject_driven_models.OmniControlModel'
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