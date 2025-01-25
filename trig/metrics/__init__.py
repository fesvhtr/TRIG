from .base import BaseMetric

AVAILABLE_Metric = {
    'base': 'BaseMetric',
    'gpt_logit': 'GPTLogitMetric',
}


def import_model(model_name):
    """Dynamically import a specific model based on name."""
    model_name = model_name.lower()
    if model_name not in AVAILABLE_Metric:
        raise ValueError(f'Model {model_name} not found in AVAILABLE_LMM')
    try:
        # Modified import statement to use the correct module path
        module_path = f'trim.metrics.{model_name}'
        module = __import__(module_path, fromlist=[AVAILABLE_Metric[model_name]])
        return getattr(module, AVAILABLE_Metric[model_name])
    except Exception as e:
        raise ImportError(f'Failed to import {model_name}. Error: {e}')
