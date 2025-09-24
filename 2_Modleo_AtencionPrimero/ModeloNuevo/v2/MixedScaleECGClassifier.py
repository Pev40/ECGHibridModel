import torch.nn as nn
import torch
from LeadEncoderAttention import LeadEncoderAttention

class MixedScaleECGClassifier(nn.Module):
    def __init__(self, input_channels, mid_channels, num_classes, dropout=0.5):
        super(MixedScaleECGClassifier, self).__init__()
        
        kernel_sizes = [5, 9, 11] # Kernels para capturar features a diferentes escalas
        
        # 1. Capas convolucionales de escala mixta
        self.mixed_conv = nn.ModuleList([
            nn.Conv1d(input_channels, mid_channels, kernel_size=ks, stride=1, padding=ks//2, bias=False)
            for ks in kernel_sizes
        ])
        
        # 2. Módulo de atención LEA
        # Se inicializa para operar sobre los canales de salida de las convoluciones
        self.lea = LeadEncoderAttention(in_channels=mid_channels)

        # 3. Bloque de procesamiento post-atención
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2)
        self.do = nn.Dropout(dropout)
        
        # 4. Clasificador final
        # La dimensión de entrada del clasificador dependerá de la longitud de la secuencia
        # después del Max Pooling. Aquí se usa AdaptiveMaxPool1d para una longitud fija.
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(mid_channels, num_classes)

    def forward(self, x):
        # x_in tiene la forma: (Batch, input_channels, Time)
        
        # Aplicamos cada convolución de la lista
        x_scales = [conv(x) for conv in self.mixed_conv]  # Lista de 3 tensores de [B, C, T]
        
        # Apilamos los resultados en una nueva dimensión (dim=2) para LEA
        # El paper lo aplica sobre los leads; aquí lo adaptamos para que funcione sobre las "escalas"
        x_stacked = torch.stack(x_scales, dim=2)  # -> [B, C, 3, T]
        
        # Aplicamos la atención LEA para re-ponderar las features
        x_att = self.lea(x_stacked)  # -> [B, C, 3, T]
        
        # Fusionamos las escalas mediante un promedio, como en tu código de ejemplo
        x_fused = x_att.mean(dim=2) # -> [B, C, T]
        
        # Pasamos por el bloque de procesamiento final
        x_processed = self.do(self.mp(self.relu(self.bn(x_fused)))) # -> [B, C, T/2]
        
        # Clasificación final
        x_pooled = self.gap(x_processed).squeeze(-1) # -> [B, C]
        output = self.fc(x_pooled) # -> [B, num_classes]
        
        return output
