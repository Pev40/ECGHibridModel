import torch
import torch.nn as nn
from .vtt_attention import VariableAttention  # Asumiendo usa esta para base attn
from einops import rearrange

class VariableDifferentialAttention(nn.Module):
    """Mejorado: Differential full como Diff Transformer, adaptado a multi-lead (intra vs inter)"""
    def __init__(self, d_model, nhead=8, dropout=0.1, chunk_size=512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn_var = VariableAttention(d_model, heads=nhead, dropout=dropout)  # Base
        self.to_qkv_intra = nn.Linear(d_model, d_model * 3, bias=False)  # Para intra-lead
        self.to_qkv_inter = nn.Linear(d_model, d_model * 3, bias=False)  # Para inter-lead (global)
        self.to_out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        # Fix: Usar detach().clone() para evitar warning
        self.lambda_diff = nn.Parameter((0.8 - 0.6 * torch.exp(torch.tensor(-0.3))).detach().clone())
        self.dropout = nn.Dropout(dropout)
        self.chunk_size = chunk_size
        
    def forward(self, x, use_attn=False):
        b, t, f, d = x.shape  # f=leads
        residual = x
        x_norm = self.norm(x)
        
        # Intra-lead attn: Atención local por lead
        x_resh = rearrange(x_norm, 'b t f d -> (b f) t d')  # Treat leads separately
        qkv_intra = self.to_qkv_intra(x_resh).chunk(3, dim=-1)
        attn_intra = self._compute_attn_chunked(*qkv_intra)  # Chunked attention
        
        # Inter-lead attn: Global (mean over leads)
        x_mean = x_norm.mean(dim=2, keepdim=True).expand_as(x_norm)  # Mean inter-lead
        x_mean_resh = rearrange(x_mean, 'b t f d -> (b f) t d')
        qkv_inter = self.to_qkv_inter(x_mean_resh).chunk(3, dim=-1)
        attn_inter = self._compute_attn_chunked(*qkv_inter)  # Chunked attention
        
        # Differential: Subtract para cancel noise
        diff_out = attn_intra - self.lambda_diff * attn_inter
        diff_out = self.to_out(diff_out)
        diff_out = rearrange(diff_out, '(b f) t d -> b t f d', f=f)
        
        # Combine con residual (como original)
        out = residual + diff_out
        out = self.ffn(out) + out
        out = self.dropout(out)
        weights = None  # Si need, compute from attn_var if use_attn
        return out, weights
    
    def _compute_attn_chunked(self, q, k, v):
        """Compute attention in chunks to reduce memory usage"""
        b, t, d = q.shape
        scale = d ** -0.5
        
        if t <= self.chunk_size:
            # Si la secuencia es pequeña, usar atención normal
            return self._compute_attn(q, k, v)
        
        # Dividir en chunks
        num_chunks = (t + self.chunk_size - 1) // self.chunk_size
        output_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, t)
            
            # Chunk actual
            q_chunk = q[:, start_idx:end_idx]  # [b, chunk_size, d]
            
            # Para cada chunk, computar atención con todos los keys/values
            # pero solo para el chunk actual de queries
            sim = torch.einsum('b i d, b j d -> b i j', q_chunk, k) * scale
            attn_weights = torch.softmax(sim, dim=-1)
            attn_output = torch.einsum('b i j, b j d -> b i d', attn_weights, v)
            
            output_chunks.append(attn_output)
        
        return torch.cat(output_chunks, dim=1)
    
    def _compute_attn(self, q, k, v):
        """Original attention computation for small sequences"""
        scale = (q.size(-1)) ** -0.5
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale
        return torch.softmax(sim, dim=-1) @ v