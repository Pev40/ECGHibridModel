import torch
import torch.nn as nn
from .vtt_attention import VariableAttention  # Asumiendo usa esta para base attn
from einops import rearrange

class VariableDifferentialAttention(nn.Module):
    """Mejorado: Differential full como Diff Transformer, adaptado a multi-lead (intra vs inter)"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn_var = VariableAttention(d_model, heads=nhead, dropout=dropout)  # Base
        self.to_qkv_intra = nn.Linear(d_model, d_model * 3, bias=False)  # Para intra-lead
        self.to_qkv_inter = nn.Linear(d_model, d_model * 3, bias=False)  # Para inter-lead (global)
        self.to_out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.lambda_diff = nn.Parameter(torch.tensor(0.8 - 0.6 * torch.exp(torch.tensor(-0.3))))  # Init como Diff Transformer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, use_attn=False):
        b, t, f, d = x.shape  # f=leads
        residual = x
        x_norm = self.norm(x)
        
        # Intra-lead attn: Atención local por lead
        x_resh = rearrange(x_norm, 'b t f d -> (b f) t d')  # Treat leads separately
        qkv_intra = self.to_qkv_intra(x_resh).chunk(3, dim=-1)
        attn_intra = self._compute_attn(*qkv_intra)  # Softmax attn
        
        # Inter-lead attn: Global (mean over leads)
        x_mean = x_norm.mean(dim=2, keepdim=True).expand_as(x_norm)  # Mean inter-lead
        x_mean_resh = rearrange(x_mean, 'b t f d -> (b f) t d')
        qkv_inter = self.to_qkv_inter(x_mean_resh).chunk(3, dim=-1)
        attn_inter = self._compute_attn(*qkv_inter)
        
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
    
    def _compute_attn(self, q, k, v):
        scale = (q.size(-1)) ** -0.5
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale
        return torch.softmax(sim, dim=-1) @ v