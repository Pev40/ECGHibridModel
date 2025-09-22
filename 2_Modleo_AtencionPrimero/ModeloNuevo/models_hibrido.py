import torch
from torch import nn

# Reutilizamos bloques del ECGTransForm
from .ecgtransform import SEBasicBlock

# Atención Variable (copia local basada en VTT)
from .vtt_attention import VariableAttention


class ECGHybridVariableBeforeBiTrans(nn.Module):
    def __init__(self, configs, hparams, num_leads=None):
        super(ECGHybridVariableBeforeBiTrans, self).__init__()

        # CNN multi-escala del modelo original
        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(configs.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs.dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # CRM
        self.inplanes = configs.final_out_channels
        self.crm = self._make_layer(SEBasicBlock, configs.final_out_channels, 3)

        # Parámetros para Atención Variable (VTT)
        self.d_model = configs.trans_dim
        self.num_heads = configs.num_heads
        # Si no se especifica num_leads, usamos el número de derivaciones si viene en configs.input_channels
        # En muchos datasets ECG multiderivación: num_leads = configs.input_channels
        self.num_leads = num_leads if num_leads is not None else getattr(configs, 'num_leads', configs.input_channels)

        # Proyección C -> F*D para construir tokens por derivación por instante
        self.proj_tokens = nn.Conv1d(configs.final_out_channels, self.num_leads * self.d_model, kernel_size=1, bias=False)

        # Atención Variable del VTT opera sobre [B, T, F, D]
        self.var_attn = VariableAttention(dim=self.d_model, heads=self.num_heads, dim_head=self.d_model, dropout=configs.dropout)

        # Bi-Transformer temporal (igual que el ECGTransForm)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # Cabezas de clasificación (coarse y fine)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.num_coarse = getattr(configs, 'num_coarse', None)
        self.num_fine = getattr(configs, 'num_fine', None)
        if (self.num_coarse is not None) and (self.num_fine is not None):
            self.head_coarse = nn.Linear(self.d_model, self.num_coarse)
            self.head_fine = nn.Linear(self.d_model, self.num_fine)
        else:
            self.head_coarse = None
            self.head_fine = None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_internal(self, x_in, request_attn=False):
        # CNN multi-escala
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)
        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # CRM
        x = self.crm(x)  # [B, C, T]

        # Proyección a tokens por derivación: [B, C, T] -> [B, F*D, T] -> [B, T, F, D]
        x_tok = self.proj_tokens(x)
        b, _, t = x_tok.shape
        x_tok = x_tok.view(b, self.num_leads, self.d_model, t)  # [B, F, D, T]
        x_tok = x_tok.permute(0, 3, 1, 2)  # [B, T, F, D]

        # Atención Variable (entre derivaciones)
        attn = None
        x_tok, attn = self.var_attn(x_tok, use_attn=request_attn)  # [B, T, F, D]

        # Pooling sobre F para BiTrans: media simple
        x_var = x_tok.mean(dim=2)  # [B, T, D]

        # Bi-Transformer temporal bidireccional
        x_fwd = self.transformer_encoder(x_var)  # [B, T, D]
        x_bwd = self.transformer_encoder(torch.flip(x_var, [1]))  # [B, T, D]
        x_bt = x_fwd + x_bwd

        # Pooling y clasificador
        x_bt = x_bt.permute(0, 2, 1)  # [B, D, T]
        x_bt = self.aap(x_bt)  # [B, D, 1]
        x_flat = x_bt.reshape(x_bt.shape[0], -1)
        if (self.head_coarse is None) or (self.head_fine is None):
            raise RuntimeError('Configure num_coarse y num_fine en configs antes de usar el modelo')
        logits_coarse = self.head_coarse(x_flat)
        logits_fine = self.head_fine(x_flat)
        return logits_coarse, logits_fine, attn

    def forward(self, x_in, use_attn=False):
        logits_coarse, logits_fine, _ = self._forward_internal(x_in, request_attn=use_attn)
        return logits_coarse, logits_fine

    def forward_with_attn(self, x_in):
        logits_coarse, logits_fine, attn = self._forward_internal(x_in, request_attn=True)
        return logits_coarse, logits_fine, attn


