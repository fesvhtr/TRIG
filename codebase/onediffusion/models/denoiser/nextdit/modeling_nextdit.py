
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Any, Tuple, Optional
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from .layers import LLamaFeedForward, RMSNorm

# import frasch


def modulate(x, scale):
    return x * (1 + scale)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=t.dtype) / half
        ).to(t.device)
        args = t[:, :, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, num_patches, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, num_patches * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), hidden_size),
        )
        
    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads=None,
        qk_norm=False,
        y_dim=0,
        base_seqlen=None,
        proportional_attn=False,
        attention_dropout=0.0,
        max_position_embeddings=384,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.qk_norm = qk_norm
        self.y_dim = y_dim
        self.base_seqlen = base_seqlen
        self.proportional_attn = proportional_attn
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        if y_dim > 0:
            self.wk_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros(n_heads))

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim, eps=1e-6)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()


    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        # xq, xk: [batch_size, seq_len, n_heads, head_dim]
        # freqs_cis: [1, seq_len, 1, head_dim]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

        xq_complex = torch.view_as_complex(xq_)
        xk_complex = torch.view_as_complex(xk_)
        
        freqs_cis = freqs_cis.unsqueeze(2)

        # Apply freqs_cis
        xq_out = xq_complex * freqs_cis
        xk_out = xk_complex * freqs_cis

        # Convert back to real numbers
        xq_out = torch.view_as_real(xq_out).flatten(-2)
        xk_out = torch.view_as_real(xk_out).flatten(-2)

        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x,
        x_mask,
        freqs_cis,
        y=None,
        y_mask=None,
        init_cache=False,
    ):
        bsz, seqlen, _ = x.size()
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        if x_mask is None:
            x_mask = torch.ones(bsz, seqlen, dtype=torch.bool, device=x.device)
        inp_dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(n_rep, dim=2)
            xv = xv.repeat_interleave(n_rep, dim=2)

        freqs_cis = freqs_cis.to(xq.device)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)

        if inp_dtype in [torch.float16, torch.bfloat16]:
            # begin var_len flash attn
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states.to(inp_dtype),
                key_states.to(inp_dtype),
                value_states.to(inp_dtype),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=None,
                softcap=30,
            )
            output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
        else:
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_heads, seqlen, -1),
                    scale=None,
                )
                .permute(0, 2, 1, 3)
                .to(inp_dtype)
            ) #ok


        if hasattr(self, "wk_y"):
            yk = self.ky_norm(self.wk_y(y)).view(bsz, -1, self.n_kv_heads, self.head_dim)
            yv = self.wv_y(y).view(bsz, -1, self.n_kv_heads, self.head_dim)
            n_rep = self.n_heads // self.n_kv_heads
            # if n_rep >= 1:
            #     yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            #     yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            if n_rep >= 1:
                yk = einops.repeat(yk, "b l h d -> b l (repeat h) d", repeat=n_rep)
                yv = einops.repeat(yv, "b l h d -> b l (repeat h) d", repeat=n_rep)
            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                y_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_heads, seqlen, -1).to(torch.bool),
            ).permute(0, 2, 1, 3)
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output = output + output_y

        output = output.flatten(-2)
        output = self.wo(output)

        return output.to(inp_dtype)

class TransformerBlock(nn.Module):
    """
    Corresponds to the Transformer block in the JAX code.
    """
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        qk_norm,
        y_dim,
        max_position_embeddings,
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim=y_dim, max_position_embeddings=max_position_embeddings)
        self.feed_forward = LLamaFeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 4 * dim),
        )
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x,
        x_mask,
        freqs_cis,
        y,
        y_mask,
        adaln_input=None,
    ):
        if adaln_input is not None:
            scales_gates = self.adaLN_modulation(adaln_input)
            # TODO: Duong - check the dimension of chunking
            # scale_msa, gate_msa, scale_mlp, gate_mlp = scales_gates.chunk(4, dim=-1)
            scale_msa, gate_msa, scale_mlp, gate_mlp = scales_gates.chunk(4, dim=-1)
            x = x + torch.tanh(gate_msa) * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa), # ok
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y), # ok
                    y_mask,
                )
            )
            x = x + torch.tanh(gate_mlp) * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )
        else:
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                )
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class NextDiT(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone for joint image-video training.
    """
    @register_to_config
    def __init__(
        self,
        input_size=(1, 32, 32),
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=4096,
        depth=32,
        num_heads=32,
        num_kv_heads=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        pred_sigma=False,
        caption_channels=4096,
        qk_norm=False,
        norm_type="rms",
        model_max_length=120,
        rotary_max_length=384,
        rotary_max_length_t=None
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.pred_sigma = pred_sigma
        self.caption_channels = caption_channels
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.model_max_length = model_max_length
        self.rotary_max_length = rotary_max_length
        self.rotary_max_length_t = rotary_max_length_t
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        self.x_embedder = nn.Linear(np.prod(self.patch_size) * in_channels, hidden_size)

        self.t_embedder = TimestepEmbedder(min(hidden_size, 1024))
        self.y_embedder = nn.Sequential(
            nn.LayerNorm(caption_channels, eps=1e-6),
            nn.Linear(caption_channels, min(hidden_size, 1024)),
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                n_heads=num_heads,
                n_kv_heads=self.num_kv_heads,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                y_dim=caption_channels,
                max_position_embeddings=rotary_max_length,
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            num_patches=np.prod(patch_size),
            out_channels=self.out_channels,
        )

        assert (hidden_size // num_heads) % 6 == 0, "3d rope needs head dim to be divisible by 6"

        self.freqs_cis = self.precompute_freqs_cis(
            hidden_size // num_heads,
            self.rotary_max_length,
            end_t=self.rotary_max_length_t
        )
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # self.freqs_cis = self.freqs_cis.to(*args, **kwargs)
        return self

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        end_t: int = None,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
    ):
        if timestep < scale_watershed:
            linear_factor = scale_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scale_factor

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim)) / linear_factor

        timestep = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis = torch.exp(1j * freqs)

        if end_t is not None:
            freqs_t = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim)) / linear_factor
            timestep_t = torch.arange(end_t, dtype=torch.float32)
            freqs_t = torch.outer(timestep_t, freqs_t).float()
            freqs_cis_t = torch.exp(1j * freqs_t)
            freqs_cis_t = freqs_cis_t.view(end_t, 1, 1, dim // 6).repeat(1, end, end, 1)
        else:
            end_t = end
            freqs_cis_t = freqs_cis.view(end_t, 1, 1, dim // 6).repeat(1, end, end, 1)
            
        freqs_cis_h = freqs_cis.view(1, end, 1, dim // 6).repeat(end_t, 1, end, 1)
        freqs_cis_w = freqs_cis.view(1, 1, end, dim // 6).repeat(end_t, end, 1, 1)
        freqs_cis = torch.cat([freqs_cis_t, freqs_cis_h, freqs_cis_w], dim=-1).view(end_t, end, end, -1)
        return freqs_cis

    def forward(
        self, 
        samples, 
        timesteps, 
        encoder_hidden_states,
        encoder_attention_mask,
        scale_factor: float = 1.0, # scale_factor for rotary embedding
        scale_watershed: float = 1.0, # scale_watershed for rotary embedding
    ):
        if samples.ndim == 4: # B C H W
            samples = samples[:, None, ...] # B F C H W
        
        precomputed_freqs_cis = None
        if scale_factor != 1 or scale_watershed != 1:
            precomputed_freqs_cis = self.precompute_freqs_cis(
                self.hidden_size // self.num_heads,
                self.rotary_max_length,
                end_t=self.rotary_max_length_t,
                scale_factor=scale_factor,
                scale_watershed=scale_watershed,
                timestep=torch.max(timesteps.cpu()).item()
            )
            
        if len(timesteps.shape) == 5:
            t, *_ = self.patchify(timesteps, precomputed_freqs_cis)
            timesteps = t.mean(dim=-1)
        elif len(timesteps.shape) == 1:
            timesteps = timesteps[:, None, None, None, None].expand_as(samples)
            t, *_ = self.patchify(timesteps, precomputed_freqs_cis)
            timesteps = t.mean(dim=-1)
        samples, T, H, W, freqs_cis = self.patchify(samples, precomputed_freqs_cis)
        samples = self.x_embedder(samples)
        t = self.t_embedder(timesteps)

        encoder_attention_mask_float = encoder_attention_mask[..., None].float()
        encoder_hidden_states_pool = (encoder_hidden_states * encoder_attention_mask_float).sum(dim=1) / (encoder_attention_mask_float.sum(dim=1) + 1e-8)
        encoder_hidden_states_pool = encoder_hidden_states_pool.to(samples.dtype)
        y = self.y_embedder(encoder_hidden_states_pool)
        y = y.unsqueeze(1).expand(-1, samples.size(1), -1)

        adaln_input = t + y
                                
        for block in self.layers:
            samples = block(samples, None, freqs_cis, encoder_hidden_states, encoder_attention_mask, adaln_input)

        samples = self.final_layer(samples, adaln_input)
        samples = self.unpatchify(samples, T, H, W)

        return samples

    def patchify(self, x, precompute_freqs_cis=None):
        # pytorch is C, H, W
        B, T, C, H, W = x.size()
        pT, pH, pW = self.patch_size
        x = x.view(B, T // pT, pT, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
        x = x.reshape(B, -1, pT * pH * pW * C)
        if precompute_freqs_cis is None:
            freqs_cis = self.freqs_cis[: T // pT, :H // pH, :W // pW].reshape(-1, * self.freqs_cis.shape[3:])[None].to(x.device)
        else:
            freqs_cis = precompute_freqs_cis[: T // pT, :H // pH, :W // pW].reshape(-1, * precompute_freqs_cis.shape[3:])[None].to(x.device)
        return x, T // pT, H // pH, W // pW, freqs_cis

    def unpatchify(self, x, T, H, W):
        B = x.size(0)
        C = self.out_channels
        pT, pH, pW = self.patch_size
        x = x.view(B, T, H, W, pT, pH, pW, C)
        x = x.permute(0, 1, 4, 7, 2, 5, 3, 6)
        x = x.reshape(B, T * pT, C, H * pH, W * pW)
        return x
