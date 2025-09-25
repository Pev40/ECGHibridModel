import torch
import torch.nn as nn
from .vtt_attention import VariableAttention


class VariableDifferentialAttention(nn.Module):
    """Twist: Variable (de VTT) + Differential (subtract para ruido lead-específico)"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn_var = VariableAttention(d_model, heads=nhead, dropout=dropout)  # Tu VTT
        self.ffn = nn.Sequential(  # Post-differential refiner
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.lambda_diff = nn.Parameter(torch.tensor(0.5))  # Learnable balance
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn=False):
        residual = x
        x_norm = self.norm(x)
        # Variable attn (cross-F/leads)
        var_out, var_weights = self.attn_var(x_norm, use_attn=use_attn)
        # Differential: Subtract mean lead para reducir ruido común
        diff_out = var_out - var_out.mean(dim=2, keepdims=True)  # [B,T,F,D] - mean_F
        # Combine + residual
        out = residual + self.lambda_diff * diff_out + (1 - self.lambda_diff) * var_out
        out = self.ffn(out) + out  # FFN residual
        out = self.dropout(out)
        weights = var_weights if use_attn else None
        return out, weights