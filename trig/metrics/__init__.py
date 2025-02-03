from .base import BaseMetric
import importlib

AVAILABLE_Metric = {
    'base': 'base.BaseMetric',
    'gpt_logit': 'gpt_logit.GPTLogitMetric',
    'gpt_text': 'gpt_text.GPTTextMetric',
    'niqe': 'niqe.NIQEMetric',
    'aes_predictor': 'aes_predictor.AESPredictorMetric',
}




def import_metric(metric_name):
    """
    Dynamically import a specific class based on its name.

    :param metric_name: The key in AVAILABLE_Metric (e.g., 'BaseMetric').
    :return: The imported class.
    """
    if metric_name not in AVAILABLE_Metric:
        raise ValueError(f"Metric {metric_name} not found in AVAILABLE_Metric.")

    try:
        full_path = f"trig.metrics.{AVAILABLE_Metric[metric_name]}"
        module_path, class_name = full_path.rsplit('.', 1)

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to import metric {metric_name}. Error: {e}")


