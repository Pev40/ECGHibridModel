import torch
import math
import torch.nn as nn
from transformers import SwinConfig, SwinModel
from .StockwellTransform import StockwellTransform
import numpy as np

# (Incluir la clase StockwellTransform definida anteriormente aquí)

class StockwellSwinEncoder(nn.Module):
    """
    Un codificador híbrido que aplica la Transformada de Stockwell y luego
    un Swin Transformer para extraer características de series temporales.
    
    Esta versión utiliza una implementación real de S-transform y maneja
    correctamente la salida compleja para el modelo de visión.
    
    Args:
        input_dim (int): Dimensión de las características de entrada (D en [B, T, D]).
        freq_bins_low (int): Frecuencia de inicio para S-transform.
        freq_bins_high (int): Frecuencia final (exclusiva) para S-transform.
        swin_model_name (str): Modelo Swin pre-entrenado de Hugging Face.
    """
    def __init__(self, input_dim: int, freq_bins_low: int = 1, freq_bins_high: int = 65, swin_model_name: str = 'microsoft/swin-tiny-patch4-window7-224'):
        super().__init__()
        self.input_dim = input_dim
        self.freq_bins_low = freq_bins_low
        self.freq_bins_high = freq_bins_high
        self.num_freq_bins = freq_bins_high - freq_bins_low
        
        # 1. Módulo de la Transformada de Stockwell (implementación real)
        self.stockwell = StockwellTransform(freq_bins_low=self.freq_bins_low, freq_bins_high=self.freq_bins_high)

        # 2. Configurar el Swin Transformer para la entrada de la S-transform
        try:
            config = SwinConfig.from_pretrained(swin_model_name)
        except Exception:
            # Fallback sin internet ni pesos preentrenados
            config = SwinConfig()
        
        # El número de canales de entrada para Swin será el doble del `input_dim`,
        # ya que apilaremos las partes real e imaginaria.
        self.swin_input_channels = self.input_dim * 2
        config.num_channels = self.swin_input_channels
        
        # La altura y el ancho deben ser divisibles por parche y ventana en todas las etapas.
        # Usamos patch=4 y window_size=2 para que el último nivel (con fuerte reducción) siga siendo válido.
        config.patch_size = (4, 4)
        config.window_size = 2
        
        self.temporal_encoder = SwinModel(config)
        print(f"Swin Transformer adaptado para aceptar {self.swin_input_channels} canales (Real + Imaginario).")

    def freeze_stages(self, num_stages: int = 1):
        """Congela las primeras `num_stages` etapas del Swin para regularización/fine-tuning estable."""
        if num_stages <= 0:
            return
        try:
            # Basado en estructura típica de Swin: embeddings + layers (stages)
            self.temporal_encoder.embeddings.eval()
            for p in self.temporal_encoder.embeddings.parameters():
                p.requires_grad = False
            stages = list(self.temporal_encoder.encoder.layers)
            for i, layer in enumerate(stages):
                if i < num_stages:
                    layer.eval()
                    for p in layer.parameters():
                        p.requires_grad = False
        except Exception:
            pass
        
    def _pad_to_multiples(self, x: torch.Tensor, cfg: SwinConfig) -> torch.Tensor:
        """Padding para que H y W sean múltiplos de patch*2^(stages-1)*window_size.
        Esto garantiza que, tras patch embed y los 3 merges, las ventanas sean válidas.
        """
        H, W = x.shape[-2:]
        patch_size = cfg.patch_size[0] if isinstance(cfg.patch_size, (tuple, list)) else int(cfg.patch_size)
        window_size = cfg.window_size if isinstance(cfg.window_size, int) else int(cfg.window_size[0])
        num_stages = len(getattr(cfg, 'depths', [2, 2, 6, 2]))
        num_merges = max(0, num_stages - 1)
        required_multiple = patch_size * (2 ** num_merges) * window_size
        pad_h = (required_multiple - (H % required_multiple)) % required_multiple
        pad_w = (required_multiple - (W % required_multiple)) % required_multiple
        if pad_h > 0 or pad_w > 0:
            return nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante.
        
        Args:
            x (torch.Tensor): Tensor de entrada [Batch, Time, D_input].
        
        Returns:
            torch.Tensor: Vector de características de salida [Batch, embed_dim].
        """
        # 0. Preparar la entrada: [B, T, D] -> [B, D, T]
        x = x.permute(0, 2, 1)

        # 1. Aplicar Transformada de Stockwell real
        # Entrada: [B, D, T] -> Salida: [B, D, num_freq_bins, T] (complejo)
        x_stockwell = self.stockwell(x)
        
        # 2. Separar y apilar partes real e imaginaria
        # Esto crea una "imagen" con el doble de canales, todos reales.
        x_real = torch.real(x_stockwell)
        x_imag = torch.imag(x_stockwell)
        # Forma -> [B, 2*D, num_freq_bins, T]
        x_image = torch.cat([x_real, x_imag], dim=1)
        
        # 3. Asegurar que las dimensiones son divisibles por el tamaño del parche
        x_padded = self._pad_to_multiples(x_image, self.temporal_encoder.config)
        
        # 4. Pasar a través del Swin Transformer
        outputs = self.temporal_encoder(x_padded)
        # Usamos mean pooling sobre los tokens (no todas las variantes exponen pooler_output)
        x_tokens = outputs.last_hidden_state  # [B, N_patches, hidden]
        x_encoded = x_tokens.mean(dim=1)
        
        return x_encoded

