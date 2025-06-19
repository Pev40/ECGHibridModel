import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
import math
from math import sqrt
import numpy as np


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), weights


class VariableAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape
        x = rearrange(x, 'b t f d -> (b t) f d')
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, '(b t) f d -> b t f d', t=t)
        return out, weights


class TemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape
        x = rearrange(x, 'b t f d -> (b f) t d')
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, '(b f) t d -> b t f d', f=f)
        return out, weights


class VariableTemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        # Reducir parámetros usando grupos compartidos
        self.variable_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.variable_to_out = nn.Linear(inner_dim, dim)
        self.variable_dropout = nn.Dropout(dropout)

        self.temporal_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.temporal_to_out = nn.Linear(inner_dim, dim)
        self.temporal_dropout = nn.Dropout(dropout)

        # Reducir dimensionalidad de salida para ahorrar memoria
        self.linear = nn.Linear(dim * 2, dim)

    def _efficient_attention(self, q, k, v, scale, dropout, use_attn):
        """Implementación más eficiente de atención para ahorrar memoria"""
        # Usar chunking para reducir uso de memoria
        chunk_size = 64  # Procesar en chunks más pequeños
        
        if q.size(2) <= chunk_size:
            # Si es pequeño, procesar directamente
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
            attn = sim.softmax(dim=-1)
            weights = attn if use_attn else None
            attn = dropout(attn)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            return out, weights
        else:
            # Procesar en chunks para ahorrar memoria
            b, h, n, d = q.shape
            out = torch.zeros_like(q)
            weights = [] if use_attn else None
            
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                q_chunk = q[:, :, i:end_i]
                
                sim = einsum('b h i d, b h j d -> b h i j', q_chunk, k) * scale
                attn = sim.softmax(dim=-1)
                
                if use_attn:
                    weights.append(attn)
                    
                attn = dropout(attn)
                out_chunk = einsum('b h i j, b h j d -> b h i d', attn, v)
                out[:, :, i:end_i] = out_chunk
            
            if use_attn:
                weights = torch.cat(weights, dim=2)
            
            return out, weights

    def forward(self, x, use_attn):
        b, t, f, d = x.shape

        # Atención Variable (más eficiente)
        vx = rearrange(x, 'b t f d -> (b t) f d')
        h = self.heads
        vq, vk, vv = self.variable_to_qkv(vx).chunk(3, dim=-1)
        vq, vk, vv = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (vq, vk, vv))
        
        vout, vweights = self._efficient_attention(vq, vk, vv, self.scale, self.variable_dropout, use_attn)
        
        vout = rearrange(vout, 'b h n d -> b n (h d)', h=h)
        vout = self.variable_to_out(vout)
        vout = rearrange(vout, '(b t) f d -> b t f d', t=t)

        # Atención Temporal (más eficiente)
        tx = rearrange(x, 'b t f d -> (b f) t d')
        tq, tk, tv = self.temporal_to_qkv(tx).chunk(3, dim=-1)
        tq, tk, tv = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (tq, tk, tv))
        
        tout, tweights = self._efficient_attention(tq, tk, tv, self.scale, self.temporal_dropout, use_attn)
        
        tout = rearrange(tout, 'b h n d -> b n (h d)', h=h)
        tout = self.temporal_to_out(tout)
        tout = rearrange(tout, '(b f) t d -> b t f d', f=f)

        # Combinar salidas
        out = torch.cat([vout, tout], dim=-1)
        out = self.linear(out)
        
        return out, vweights, tweights