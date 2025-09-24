import torch
import torch.nn as nn

class LeadEncoderAttention(nn.Module):
    """
    Implementación del módulo Lead Encoder Attention (LEA) descrito en el paper.
    Este módulo está diseñado para operar sobre un feature map 4D (B, C, H, W)
    y generar pesos de atención a lo largo de las dimensiones H y W.
    """
    def __init__(self, in_channels, reduction_ratio=4):
        super(LeadEncoderAttention, self).__init__()
        
        # Capa convolucional compartida para procesar las features agrupadas
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True)
        )
        
        # Convoluciones separadas para generar los mapas de atención para cada dimensión
        self.conv_h = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x tiene la forma: (Batch, Channels, Height, Width)
        # En tu caso: (Batch, mid_channels, Num_Scales, Time)
        b, c, h, w = x.size()
        
        # 1. Agregación de features a lo largo de las dimensiones H y W 
        # Pool a lo largo de la dimensión de tiempo (W)
        pooled_h = x.mean(dim=3, keepdim=True) # -> (b, c, h, 1)
        # Pool a lo largo de la dimensión de escalas/leads (H)
        pooled_w = x.mean(dim=2, keepdim=True) # -> (b, c, 1, w)

        # 2. Concatenación y transformación compartida
        # Se necesita unificar las dimensiones para concatenar y procesar.
        # Transponemos el pool de W para que tenga la misma orientación que el de H.
        pooled_w_transposed = pooled_w.permute(0, 1, 3, 2) # -> (b, c, w, 1)
        
        # Concatenamos a lo largo de la dimensión H "extendida"
        concat_features = torch.cat([pooled_h, pooled_w_transposed], dim=2) # -> (b, c, h+w, 1)
        
        # Aplicamos la transformación compartida
        transformed_features = self.shared_conv(concat_features) # -> (b, c//r, h+w, 1)
        
        # 3. Separamos para generar los pesos de atención
        f_h, f_w = torch.split(transformed_features, [h, w], dim=2) # f_h: (b, c//r, h, 1), f_w: (b, c//r, w, 1)
        
        # Revertimos la transposición de f_w
        f_w_restored = f_w.permute(0, 1, 3, 2) # -> (b, c//r, 1, w)
        
        # Generamos los mapas de atención con sigmoide
        attn_h = self.sigmoid(self.conv_h(f_h)) # -> (b, c, h, 1)
        attn_w = self.sigmoid(self.conv_w(f_w_restored)) # -> (b, c, 1, w)
        
        # 4. Re-ponderamos el feature map de entrada
        # Los mapas de atención se expandirán (broadcasting) para coincidir con el tamaño de x
        output = x * attn_h * attn_w
        
        return output