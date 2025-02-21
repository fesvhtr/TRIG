from .base import BaseMetric
import importlib

AVAILABLE_Metric = {
    'base': 'base.BaseMetric',
    'gpt_logit': 'gpt_logit.GPTLogitMetric',
    'gpt_text': 'gpt_text.GPTTextMetric',
    'niqe': 'classic.NIQEMetric',
    'aes_predictor': 'aesthetic_predictor.AESPredictorMetric',
    'knn_rarity': 'classic.KNN_rarityscore',
    'cmmd': 'classic.CMMD',
    'rt_semsr': 'classic.RT_SemSR',
    'mid': 'classic.MIDMetric',
    'tas_artscore': 'classic.TASArtScoreMetric',
}




def import_metric(metric_name):
    """
    动态导入特定的度量类。

    Args:
        metric_name: AVAILABLE_Metric中的键名（例如 'BaseMetric'）
    Returns:
        导入的类
    """
    if metric_name not in AVAILABLE_Metric:
        raise ValueError(f"度量 {metric_name} 在 AVAILABLE_Metric 中未找到。")

    try:
        full_path = f"trig.metrics.{AVAILABLE_Metric[metric_name]}"
        module_path, class_name = full_path.rsplit('.', 1)

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"导入度量 {metric_name} 失败。错误: {e}")


