import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedSpatialAttention(nn.Module):
    """
    Atención Espacial Guiada con Agrupación de Derivaciones (Group Attention).

    Este módulo genera una máscara de atención espacial (temporal) para cada grupo 
    de derivaciones, permitiendo que el modelo se enfoque en segmentos relevantes de la señal.
    La guía proviene de una capa convolucional que aprende a generar un "mapa de activación"
    similar a CAM.
    """
    def __init__(self, channels, num_groups=2):
        super(GuidedSpatialAttention, self).__init__()
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        
        if channels % num_groups != 0:
            raise ValueError("El número de canales debe ser divisible por el número de grupos.")

        # Capa convolucional "guía" que genera el mapa de atención espacial.
        # Reduce la dimensionalidad del canal para crear un único mapa de atención por grupo.
        self.attention_generator = nn.Sequential(
            nn.Conv1d(self.group_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor de entrada con forma (N, C, L), donde N es el tamaño del batch,
                        C es el número de canales (derivaciones), y L es la longitud de la señal.
        
        Returns:
            Tensor: Tensor de salida con atención aplicada, con la misma forma que la entrada.
        """
        N, C, L = x.shape
        
        # Dividir la entrada en grupos a lo largo de la dimensión de canales
        # Por ejemplo, Grupo 1: Derivaciones de extremidades, Grupo 2: Precordiales
        grouped_x = x.view(N, self.num_groups, self.group_channels, L)
        
        outputs = []
        for i in range(self.num_groups):
            group = grouped_x[:, i, :, :]  # Selecciona el i-ésimo grupo
            
            # Generar el mapa de atención espacial para este grupo.
            # La forma será (N, 1, L)
            spatial_attention_map = self.attention_generator(group)
            
            # Aplicar la máscara de atención al grupo (multiplicación elemento a elemento)
            # El broadcasting se encarga de expandir la máscara a todos los canales del grupo.
            attended_group = group * spatial_attention_map
            outputs.append(attended_group)
            
        # Concatenar los grupos atendidos para reconstruir el tensor de salida
        out = torch.cat(outputs, dim=1)
        
        return out