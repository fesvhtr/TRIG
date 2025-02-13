from trig.metrics.base import BaseMetric
from trig.metrics.niqe import calculate_niqe
import numpy as np
import cv2
class NIQEMetric(BaseMetric):
    # Natural Image Quality Evaluator
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, image_path, prompt):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"ERROR: {image_path}")
        niqe_score = calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y')
        return float(niqe_score)

    def compute_batch(self, images, prompts, dimension):
        scores = []
        for img_path in images:
            try:
                score = self.compute(img_path, None)
                scores.append(score)
            except Exception as e:
                print(f"处理图片时出错 {img_path}: {str(e)}")
                scores.append(np.nan)
        
        return np.array(scores)




