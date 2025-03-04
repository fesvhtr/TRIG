# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding models used in the CMMD calculation."""

import jax
from scenic.projects.baselines.clip import model as clipmodel
import torch
import clip
from PIL import Image
import torchvision.transforms as T

Array = jax.numpy.ndarray

_CLIP_MODEL_NAME = 'vit_l14_336px'


def _clip_preprocess(images, size):
  target_shape = images.shape[:-3] + (size, size, images.shape[-1])

  images = jax.image.resize(images, shape=target_shape, method='bicubic')

  # Apply CLIP-specific shifting/scaling.  The input to `normalize_image` is
  # expected to be in [0, 1].
  images = clipmodel.normalize_image(images)

  return images


class ClipEmbeddingModel:
  """CLIP图像嵌入计算器"""
  
  def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
    self.device = device
    self.model, _ = clip.load("ViT-L/14@336px", device=self.device)
    self.model.eval()
    self.preprocess = T.Compose([
        T.Resize(336, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(336),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                   (0.26862954, 0.26130258, 0.27577711))
    ])
    
  @torch.no_grad()
  def embed(self, images):
    """计算给定图像的CLIP嵌入
    
    Args:
        images: PIL图像列表或单个PIL图像
        
    Returns:
        torch.Tensor: 形状为(batch_size, embedding_dim)的嵌入向量
    """
    if not isinstance(images, list):
        images = [images]
        
    processed = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        processed.append(self.preprocess(img).unsqueeze(0))
        
    batch = torch.cat(processed, dim=0).to(self.device)
    return self.model.encode_image(batch)
