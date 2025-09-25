import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .VariableDifferentialAttention import VariableDifferentialAttention
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class HMST(nn.Module):
    """Hierarchical Multi-Scale Transformer para ECG multi-label jerárquico con SNOMED"""
    def __init__(self, input_channels=12+3, d_model=512, nhead=8, num_layers=6, num_stages=3,
                 num_coarse=10, num_fine=27, dropout=0.1, snomed_dim=10):
        super().__init__()
        self.d_model = d_model
        self.num_stages = num_stages
        self.input_proj = nn.Linear(input_channels, d_model)  # Proyección de (leads + wide_feats) a d_model

        # Etapas convolucionales multi-escala que preservan T y canales (=d_model)
        self.conv_stages = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model,
                      kernel_size=3 + 2 * i, stride=1, padding='same', groups=d_model)  # depthwise
            for i in range(num_stages)
        ])
        self.conv_pw = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
            for _ in range(num_stages)
        ])

        # Atención variable/diferencial por etapa
        self.attn_stages = nn.ModuleList([
            VariableDifferentialAttention(d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_stages)
        ])

        # Transformer core para fusión de etapas
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, max(1, num_layers - num_stages))

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.snomed_proj = nn.Linear(snomed_dim, d_model)  # Embed SNOMED one-hot

        # Proyección de fusión (concat de etapas → d_model)
        self.fuse_linear = nn.Linear(num_stages * d_model, d_model)

        # Cabeceras (logits, sin sigmoid)
        self.head_coarse = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_coarse))
        self.head_fine = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_fine))

        # Matriz jerarquía learnable [fine, coarse] para consistency
        self.hier_matrix = nn.Parameter(torch.randn(num_fine, num_coarse))

    def forward(self, x, wide_feats=None, snomed_embed=None, use_attn=False):
        # x: [B, 12, T]; wide_feats: [B, 3]; snomed_embed: [B, snomed_dim]
        b, _, t = x.shape
        if wide_feats is None:
            wide_feats = torch.zeros(b, 3, device=x.device, dtype=x.dtype)
        if snomed_embed is None:
            snomed_embed = torch.zeros(b, self.snomed_proj.in_features, device=x.device, dtype=x.dtype)

        # Concatenar features anchas expandiendo en T
        wide_exp = wide_feats.unsqueeze(-1).expand(-1, -1, t)  # [B,3,T]
        inp = torch.cat([x, wide_exp], dim=1)  # [B, 12+3, T]

        # Proyección a d_model en el eje de características (T como secuencia)
        inp = inp.permute(0, 2, 1)  # [B, T, C]
        inp = self.input_proj(inp)  # [B, T, d_model]

        # Conv1d depthwise + pointwise por etapa (preserva T y d)
        stages_out = []
        attn_weights = []
        x_feat = inp.permute(0, 2, 1)  # [B, d_model, T] para Conv1d
        for i in range(self.num_stages):
            stage = self.conv_stages[i](x_feat)            # [B, d_model, T]
            stage = self.conv_pw[i](stage)                 # [B, d_model, T]
            stage = stage.permute(0, 2, 1)                 # [B, T, d_model]
            stage_resh = rearrange(stage, 'b t d -> b t 1 d')
            attn_stage, w = self.attn_stages[i](stage_resh, use_attn=use_attn)
            if use_attn:
                attn_weights.append(w)
            stage_out = attn_stage.mean(dim=2)             # [B, T, d_model]
            stages_out.append(stage_out)

        # Fusión y proyección a d_model
        x_fused = torch.cat(stages_out, dim=-1)            # [B, T, num_stages*d_model]
        x_fused = self.fuse_linear(x_fused)                # [B, T, d_model]

        # Transformer con token CLS modulado por SNOMED
        cls = self.cls_token.expand(b, -1, -1) + self.snomed_proj(snomed_embed).unsqueeze(1)
        x_seq = torch.cat([cls, x_fused], dim=1)           # [B, 1+T, d_model]
        x_seq = self.transformer(x_seq)
        cls_out = x_seq[:, 0]

        # Logits
        coarse_logits = self.head_coarse(cls_out)
        fine_logits = self.head_fine(cls_out)
        return coarse_logits, fine_logits, (attn_weights if use_attn else None)

    def consistency_loss(self, coarse_pred_probs, fine_pred_probs):
        # Penaliza: fine @ hier_matrix ≈ coarse (en espacio de probabilidades)
        pred_coarse_from_fine = torch.matmul(fine_pred_probs, self.hier_matrix)
        return F.mse_loss(pred_coarse_from_fine, coarse_pred_probs)
