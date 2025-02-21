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

"""IO utilities."""

import os
from cmmd import embedding
import jax
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm


def _get_image_list(path):
  ext_list = ['png', 'jpg', 'jpeg']
  image_list = []
  for ext in ext_list:
    image_list.extend(tf.io.gfile.glob(os.path.join(path, f'*{ext}')))
    image_list.extend(tf.io.gfile.glob(os.path.join(path, f'*.{ext.upper()}')))
  # Sort the list to ensure a deterministic output.
  image_list.sort()
  return image_list


def _center_crop_and_resize(im, size):
  w, h = im.size
  l = min(w, h)
  top = (h - l) // 2
  left = (w - l) // 2
  box = (left, top, left + l, top + l)
  im = im.crop(box)
  # Note that the following performs anti-aliasing as well.
  return im.resize((size, size), resample=Image.BICUBIC)  # pytype: disable=module-attr


def _read_image(path, reshape_to):
  with tf.io.gfile.GFile(path, 'rb') as f:
    im = Image.open(f)
    im.load()

  if reshape_to > 0:
    im = _center_crop_and_resize(im, reshape_to)

  return np.asarray(im).astype(np.float32)


def _get_img_generator_fn(path, reshape_to, max_count=-1):
  """Returns a generator function that yields one image at a time.

  Args:
    path: Directory to read .jpg and .png imges from.
    reshape_to: If positive, reshape images to a square images of this size.
    max_count: The maximum number of images to read.

  Returns:
    A generation function that yields images.
  """
  img_path_list = _get_image_list(path)
  if max_count > 0:
    img_path_list = img_path_list[:max_count]

  def gen():
    for img_path in img_path_list:
      x = _read_image(img_path, reshape_to)
      if x.ndim == 3:
        yield x
      elif x.ndim == 2:
        # Convert grayscale to RGB by duplicating the channel dimension.
        yield np.tile(x[Ellipsis, np.newaxis], (1, 1, 3))
      else:
        raise ValueError(
            f'Image has {x.ndim} dimensions, which is not supported. Only '
            'images with 1 or 3 color channels are currently supported.'
        )

  return gen, len(img_path_list)


def compute_embeddings_for_dir(directory, embedding_model, batch_size=32, max_count=-1):
    """计算目录中所有图像的嵌入向量
    
    Args:
        directory: 包含图像的目录路径
        embedding_model: ClipEmbeddingModel实例
        batch_size: 批处理大小
        max_count: 最大处理图像数量，-1表示处理所有图像
        
    Returns:
        torch.Tensor: 所有图像的嵌入向量
    """
    # 获取所有图像文件
    image_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, f))
    
    if max_count > 0:
        image_files = image_files[:max_count]
    
    if not image_files:
        raise ValueError(f"在目录 {directory} 中没有找到有效的图像文件")
    
    # 批量处理图像
    all_embeddings = []
    for i in tqdm(range(0, len(image_files), batch_size), desc="计算图像嵌入"):
        batch_files = image_files[i:i + batch_size]
        try:
            batch_embeddings = embedding_model.embed(batch_files)
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            continue
    
    if not all_embeddings:
        raise ValueError("没有成功处理任何图像")
    
    import torch
    return torch.cat(all_embeddings, dim=0)
