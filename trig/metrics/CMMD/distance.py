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

"""A memory-efficient MMD implementation in JAX."""

import jax
import jax.numpy as jnp
import torch

Array = jnp.ndarray

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


@jax.jit
def mmd(x, y):
  """A memory-efficient MMD implementation in JAX.

  This implements the minimum-variance/biased version of the estimator described
  in Eq.(5) of
  https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
  As described in Lemma 6's proof in that paper, the unbiased estimate and the
  minimum-variance estimate for MMD are almost identical.

  Note that the first invocation of this function will be considerably slow due
  to JAX JIT compilation.

  Args:
    x: The first set of embeddings of shape (n, embedding_dim).
    y: The second set of embeddings of shape (n, embedding_dim).

  Returns:
    The MMD distance between x and y embedding sets.
  """
  x = jnp.asarray(x)
  y = jnp.asarray(y)

  # jnp.matmul(x, x.T) etc. are not cached to avoid OOM when x has many rows.
  x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
  y_sqnorms = jnp.diag(jnp.matmul(y, y.T))

  gamma = 1 / (2 * _SIGMA**2)
  k_xx = jnp.mean(
      jnp.exp(
          -gamma
          * (
              -2 * jnp.matmul(x, x.T)
              + jnp.expand_dims(x_sqnorms, 1)
              + jnp.expand_dims(x_sqnorms, 0)
          )
      )
  )
  k_xy = jnp.mean(
      jnp.exp(
          -gamma
          * (
              -2 * jnp.matmul(x, y.T)
              + jnp.expand_dims(x_sqnorms, 1)
              + jnp.expand_dims(y_sqnorms, 0)
          )
      )
  )
  k_yy = jnp.mean(
      jnp.exp(
          -gamma
          * (
              -2 * jnp.matmul(y, y.T)
              + jnp.expand_dims(y_sqnorms, 1)
              + jnp.expand_dims(y_sqnorms, 0)
          )
      )
  )

  return _SCALE * (k_xx + k_yy - 2 * k_xy)

def mmd(ref_embs, eval_embs, sigma=1.0):
    """计算两组嵌入之间的MMD（Maximum Mean Discrepancy）距离
    
    Args:
        ref_embs: 参考图像的嵌入向量，形状为(N, D)
        eval_embs: 评估图像的嵌入向量，形状为(M, D)
        sigma: RBF核的带宽参数
        
    Returns:
        float: MMD距离值
    """
    if isinstance(ref_embs, torch.Tensor):
        ref_embs = ref_embs.detach().cpu().numpy()
    if isinstance(eval_embs, torch.Tensor):
        eval_embs = eval_embs.detach().cpu().numpy()
        
    ref_embs = torch.from_numpy(ref_embs)
    eval_embs = torch.from_numpy(eval_embs)
    
    # 计算核矩阵
    def rbf_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist**2 / (2 * sigma**2))
    
    k_xx = rbf_kernel(ref_embs, ref_embs, sigma).mean()
    k_yy = rbf_kernel(eval_embs, eval_embs, sigma).mean()
    k_xy = rbf_kernel(ref_embs, eval_embs, sigma).mean()
    
    # 计算MMD
    mmd = k_xx + k_yy - 2 * k_xy
    return float(mmd * 1000)  # 缩放因子