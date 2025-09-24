import math
import torch
import torch.nn as nn

from .LeadEncoderAttention import LeadEncoderAttention
from .GuidedSpatialAttention import GuidedSpatialAttention
from .HierarchicalVariableAttention import HierarchicalVariableAttention
from .StockwellSwinEncoder import StockwellSwinEncoder


class MixedScaleLEAFeatureExtractor(nn.Module):
    """
    Extracción de características con convoluciones multi-escala + LEA (atención sobre escalas) + post-procesamiento.
    Entrada:  x [B, C_in, T]
    Salida:   feat [B, C_mid, T_out]
    """
    def __init__(self, input_channels: int, mid_channels: int, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        kernel_sizes = [5, 9, 11]
        self.mixed_conv = nn.ModuleList([
            nn.Conv1d(input_channels, mid_channels, kernel_size=ks, stride=stride, padding=ks // 2, bias=False)
            for ks in kernel_sizes
        ])
        self.lea = LeadEncoderAttention(in_channels=mid_channels)
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool1d(kernel_size=2)
        self.do = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scales = [conv(x) for conv in self.mixed_conv]  # list of [B, C_mid, T]
        x_stack = torch.stack(x_scales, dim=2)            # [B, C_mid, 3, T]
        x_stack = self.lea(x_stack)                       # [B, C_mid, 3, T]
        x_fused = x_stack.mean(dim=2)                     # [B, C_mid, T]
        x_out = self.do(self.mp(self.relu(self.bn(x_fused))))
        return x_out                                      # [B, C_mid, T/2]


class ECGHybridVariableBeforeBiTransV2(nn.Module):
    """
    v2: CNN multi-escala + LEA -> GSA -> Variable Attention Jerárquica (MIL) -> Swin+Stockwell -> Heads jerárquicas.
    """
    def __init__(self, configs, meta=None):
        super().__init__()
        self.configs = configs
        self.num_leads = int(getattr(configs, 'num_leads', getattr(configs, 'input_channels', 12)))
        self.num_fine = int(getattr(configs, 'num_fine', 1))
        self.num_coarse = int(getattr(configs, 'num_coarse', 1))

        mid_channels = int(getattr(configs, 'mid_channels', 32))
        trans_dim = int(getattr(configs, 'trans_dim', 32))
        num_heads = int(getattr(configs, 'num_heads', 4))

        dropout = float(getattr(configs, 'dropout', 0.3))
        attn_dropout = float(getattr(configs, 'attn_dropout', dropout))
        stride = int(getattr(configs, 'stride', 1))

        # 1) CNN Multi-escala + LEA
        self.feature_extractor = MixedScaleLEAFeatureExtractor(
            input_channels=int(getattr(configs, 'input_channels', 12)),
            mid_channels=mid_channels,
            stride=stride,
            dropout=dropout,
        )

        # 2) Guided Spatial Attention (sobre canales)
        # Se asume 2 grupos para simular agrupación de derivaciones (extremidades vs precordiales)
        self.gsa = GuidedSpatialAttention(channels=mid_channels, num_groups=2 if mid_channels % 2 == 0 else 1)

        # 3) Proyección a espacio del Transformer (temporal tokens)
        self.proj_to_trans = nn.Conv1d(mid_channels, trans_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(trans_dim)
        self.proj_act = nn.GELU()

        # 4) Variable Attention Jerárquica (MIL): trabaja con tokens de forma [B, T_seg, F_inst, D]
        self.var_attn = HierarchicalVariableAttention(
            dim=trans_dim,
            heads=num_heads,
            dim_head=max(8, trans_dim // max(1, num_heads)),
            dropout=attn_dropout,
            pooling_factor=2,
            num_levels=2,
            mil=True,
        )

        # 5) Encoder temporal basado en Stockwell + Swin
        freq_low = int(getattr(configs, 'stockwell_freq_low', 1))
        freq_high = int(getattr(configs, 'stockwell_freq_high', 65))
        self.temporal_encoder = StockwellSwinEncoder(input_dim=trans_dim, freq_bins_low=freq_low, freq_bins_high=freq_high)
        freeze_stages = int(getattr(configs, 'swin_freeze_stages', 0))
        if hasattr(self.temporal_encoder, 'freeze_stages') and freeze_stages > 0:
            try:
                self.temporal_encoder.freeze_stages(freeze_stages)
            except Exception:
                pass

        # Adaptar dimensión de salida del Swin a un embedding común
        swin_embed_dim = int(getattr(self.temporal_encoder.temporal_encoder.config, 'hidden_sizes', [])[ -1 ] if hasattr(self.temporal_encoder.temporal_encoder.config, 'hidden_sizes') else getattr(self.temporal_encoder.temporal_encoder.config, 'hidden_size', trans_dim))
        self.head_embed = nn.Linear(swin_embed_dim, trans_dim)

        # Heads jerárquicas
        self.head_coarse = nn.Sequential(
            nn.LayerNorm(trans_dim),
            nn.Linear(trans_dim, self.num_coarse)
        )
        self.head_fine = nn.Sequential(
            nn.LayerNorm(trans_dim),
            nn.Linear(trans_dim, self.num_fine)
        )

    @staticmethod
    def _segment_tokens(x_td: torch.Tensor, token_size: int) -> torch.Tensor:
        """
        x_td: [B, T, D]
        Devuelve: [B, T_seg, F_inst, D] con T_seg = floor(T / token_size), F_inst = token_size.
        """
        b, t, d = x_td.shape
        if t < token_size:
            # zero-pad para alcanzar al menos un token
            pad = token_size - t
            x_td = nn.functional.pad(x_td, (0, 0, 0, pad))
            t = token_size
        t_seg = t // token_size
        t_keep = t_seg * token_size
        x_td = x_td[:, :t_keep, :]
        x_tok = x_td.view(b, t_seg, token_size, d)
        return x_tok

    def _encode_sequence(self, x_feat: torch.Tensor):
        """
        x_feat: [B, C_mid, T'] -> devuelve (x_vec, x_coarse_vec, x_fine_vec)
        """
        # Proyección a D y permutar a [B, T, D]
        x_proj = self.proj_act(self.proj_bn(self.proj_to_trans(x_feat)))  # [B, D, T']
        x_td = x_proj.permute(0, 2, 1)                                    # [B, T', D]

        # Tokenización jerárquica (coarse/fine) para Variable Attention
        # Tamaño de instancia (ventana temporal). Elegimos 32 como valor por defecto robusto.
        token_size = 32
        x_tok = self._segment_tokens(x_td, token_size=token_size)          # [B, T_seg, 32, D]

        # Atención jerárquica
        x_tok_coarse, x_tok_fine = self.var_attn(x_tok, hierarchical=True)
        # MIL pooling en cada nivel
        x_coarse_vec = x_tok_coarse.mean(dim=2)                            # [B, T_seg, D]
        x_fine_vec = x_tok_fine.mean(dim=2)                                # [B, T_seg, D]

        # Fusión de niveles y codificación temporal con Swin+Stockwell
        x_var = 0.5 * (x_coarse_vec + x_fine_vec)                           # [B, T_seg, D]
        x_enc = self.temporal_encoder(x_var)                                # [B, E]
        x_enc = self.head_embed(x_enc)                                      # [B, D]
        return x_enc, x_coarse_vec, x_fine_vec

    def forward(self, x: torch.Tensor):
        # x: [B, 12, T]
        x_feat = self.feature_extractor(x)            # [B, C_mid, T/2]
        x_feat = self.gsa(x_feat)                     # [B, C_mid, T/2]
        x_vec, _, _ = self._encode_sequence(x_feat)
        logits_coarse = self.head_coarse(x_vec)       # [B, num_coarse]
        logits_fine = self.head_fine(x_vec)           # [B, num_fine]
        return logits_coarse, logits_fine

    def forward_with_aux(self, x: torch.Tensor):
        """Devuelve logits y features intermedios para pérdidas auxiliares."""
        x_feat = self.feature_extractor(x)
        x_feat = self.gsa(x_feat)
        x_vec, x_coarse_vec, x_fine_vec = self._encode_sequence(x_feat)
        logits_coarse = self.head_coarse(x_vec)
        logits_fine = self.head_fine(x_vec)
        return logits_coarse, logits_fine, x_coarse_vec, x_fine_vec

    def forward_with_attn(self, x: torch.Tensor):
        # Compatibilidad con rutina de evaluación; mapas de atención no expuestos en esta versión
        logits_coarse, logits_fine = self.forward(x)
        attn = None
        return logits_coarse, logits_fine, attn


