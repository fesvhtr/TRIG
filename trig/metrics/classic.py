from trig.metrics.base import BaseMetric
class NIQEMetric(BaseMetric):
    # Natural Image Quality Evaluator
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, image_path, prompt):
        pass

    def compute_batch(self, images, prompts, dimension):
        pass




