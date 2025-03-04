import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import clip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 如果是多 GPU 服务器，这里选择 GPU 编号


class AestheticPredictor:
    def __init__(self, **kwargs):
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        model_weights = torch.load(r"TRIG/trig/utils/sac+logos+ava1-l14-linearMSE.pth",map_location=torch.device('cpu'))
        self.model.load_state_dict(model_weights)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def compute(self, img_path):
        pil_image = Image.open(img_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            im_emb_arr = normalized(image_features.cpu().numpy())

        with torch.no_grad():
            prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).float())

        return prediction.item()

    def compute_batch(self, data_ids, images, prompts):
        results = {}
        for data_id, image_path in zip(data_ids, images):
            results[data_id] = self.compute(image_path)
        return results


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


if __name__ == "__main__":
    aesthetic_predictor = AestheticPredictor()
    img_path = r"H:\ProjectsPro\TRIG\demo.jpg"
    score = aesthetic_predictor.compute(img_path)
    print(score)
