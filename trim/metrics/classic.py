from trim.metrics.base import BaseMetric
class FIDMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_name = "FID"

    def compute(self, image_path, prompt):
        pass

    def compute_batch(self, images, prompts, dimension):
        pass

    def compute_batch(self, images, prompts, dimension):
        pass