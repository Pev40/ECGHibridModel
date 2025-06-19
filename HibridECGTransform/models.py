import torch
from torch import nn
from einops import rearrange, repeat
from layers.Embed import PositionalEmbedding, CausalConv1d
from layers.Attention import VariableTemporalAttention
from layers.Transformer_Enc import PreNorm

class ecgTransForm(nn.Module):
    def __init__(self, configs, hparams):
        super(ecgTransForm, self).__init__()

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
        
        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        # Parámetros para la atención variable temporal
        self.embedding_dims = getattr(configs, 'embedding_dims', 256)
        self.num_transformer_blocks = getattr(configs, 'num_transformer_blocks', 3)
        self.num_heads = getattr(configs, 'num_heads', 8)
        self.attn_dropout = getattr(configs, 'attn_dropout', 0.1)
        self.ff_dropout = getattr(configs, 'ff_dropout', 0.1)
        self.time_emb = getattr(configs, 'time_emb', 4)

        # Embeddings para atención variable temporal
        self.position_embedding = PositionalEmbedding(d_model=self.embedding_dims)
        self.value_embedding = nn.Linear(self.time_emb, self.embedding_dims)

        # Convoluciones causales multi-escala
        self.causal_conv1 = CausalConv1d(configs.final_out_channels,
                                         configs.final_out_channels,
                                         kernel_size=4,
                                         dilation=1,
                                         groups=configs.final_out_channels)
        self.causal_conv2 = CausalConv1d(configs.final_out_channels,
                                         configs.final_out_channels,
                                         kernel_size=8,
                                         dilation=2,
                                         groups=configs.final_out_channels)
        self.causal_conv3 = CausalConv1d(configs.final_out_channels,
                                         configs.final_out_channels,
                                         kernel_size=16,
                                         dilation=3,
                                         groups=configs.final_out_channels)

        # Capas de transformador con atención variable temporal
        self.transformer_layers = nn.ModuleList([])
        for _ in range(self.num_transformer_blocks):
            self.transformer_layers.append(
                PreNorm(self.embedding_dims, 
                       VariableTemporalAttention(self.embedding_dims,
                                                heads=self.num_heads,
                                                dim_head=self.embedding_dims // self.num_heads,
                                                dropout=self.attn_dropout))
            )

        self.dropout = nn.Dropout(self.ff_dropout)

        # Capas finales
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(hparams["feature_dim"], configs.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
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

    def forward(self, x_in, use_attn=False):
        # Multi-scale Convolutions
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)

        # Preparar datos para atención variable temporal
        b, f, w = x.shape
        
        # Convoluciones causales multi-escala
        conv1 = self.causal_conv1(x)
        conv2 = self.causal_conv2(x)
        conv3 = self.causal_conv3(x)

        # Combinar características multi-escala
        x_multiscale = torch.stack([x, conv1, conv2, conv3], dim=-1)
        x_multiscale = rearrange(x_multiscale, 'b f w d -> b w f d')
        
        # Aplicar embeddings
        x_embedded = self.value_embedding(x_multiscale)
        
        # Añadir embeddings posicionales
        position_emb = self.position_embedding(x_embedded)
        position_emb = repeat(position_emb, 'b t d -> b t f d', f=f)
        x_embedded += position_emb
        x_embedded = self.dropout(x_embedded)

        # Aplicar capas de atención variable temporal
        variable_attn_weights = []
        temporal_attn_weights = []
        h = x_embedded
        
        for attn in self.transformer_layers:
            attn_output, vweights, tweights = attn(h, use_attn=use_attn)
            h = attn_output + h
            variable_attn_weights.append(vweights)
            temporal_attn_weights.append(tweights)

        # Preparar salida para clasificación
        h = rearrange(h, 'b w f d -> b f (w d)')
        x = self.aap(h)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        
        if use_attn:
            return x_out, [variable_attn_weights, temporal_attn_weights]
        else:
            return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
