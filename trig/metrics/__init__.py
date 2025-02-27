from .base import BaseMetric
import importlib

AVAILABLE_Metric = {
    'base': 'base.BaseMetric',
    # TRIG Metrics
    'trig_gpt': 'trig_gpt.TRIGGPTMetric',
    'trig_api': 'trig_api.TRIGAPIMetric',
    'trig_qwen_vl': 'trig_qwen_vl.TRIGQwenMetric',
    # VQAScore Metrics
    'vqascore_llava': 'vqascore_llava.VQAScoreLLaVAMetric',
    'vqascore_gpt': 'vqascore_gpt.VQAScoreGPTMetric',
    # classic Metrics
    'niqe': 'classic.NIQEMetric',
    'aes_predictor': 'aesthetic_predictor.AESPredictorMetric',
    'knn_rarity': 'classic.KNN_rarityscore',
    'rt_semsr': 'classic.RT_SemSR',
    'mid': 'classic.MIDMetric',
    'tas_artscore': 'classic.TASArtScoreMetric',
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

