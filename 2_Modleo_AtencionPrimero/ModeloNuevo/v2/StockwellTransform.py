import torch
import torch.nn as nn
import numpy as np

class StockwellTransform(nn.Module):
    """
    Implementación real y vectorizada de la Transformada de Stockwell (S-transform) en PyTorch.
    
    Esta capa transforma una señal 1D del dominio del tiempo a una representación 2D 
    compleja en el dominio tiempo-frecuencia.
    
    Args:
        freq_bins_low (int): El índice del primer bin de frecuencia a calcular. Se omite el bin 0 (DC).
        freq_bins_high (int): El índice del último bin de frecuencia a calcular.
                              El número total de bins de frecuencia será (freq_bins_high - freq_bins_low).
    """
    def __init__(self, freq_bins_low: int, freq_bins_high: int):
        super().__init__()
        if freq_bins_low < 1:
            raise ValueError("freq_bins_low debe ser 1 o mayor para evitar el componente DC.")
        if freq_bins_high <= freq_bins_low:
            raise ValueError("freq_bins_high debe ser mayor que freq_bins_low.")
            
        self.freq_bins_low = freq_bins_low
        self.freq_bins_high = freq_bins_high
        self.num_freq_bins = freq_bins_high - freq_bins_low
        print(f"StockwellTransform (Real) inicializado para calcular los bins de frecuencia del {freq_bins_low} al {freq_bins_high-1}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la S-transform de la señal de entrada.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma [Batch, Channels, Time].
        
        Returns:
            torch.Tensor: Tensor complejo con forma [Batch, Channels, num_freq_bins, Time].
        """
        B, C, T = x.shape
        
        # 1. Calcular el espectro de la señal usando FFT
        # Forma: [B, C, T] -> [B, C, T] (complejo)
        signal_fft = torch.fft.fft(x, dim=-1)
        
        # 2. Preparar frecuencias para las ventanas Gaussianas
        # Frecuencias para la ventana Gaussiana (índices m)
        freqs = torch.arange(T, device=x.device)
        
        # Frecuencias objetivo de la S-transform (índices n)
        target_freqs = torch.arange(
            self.freq_bins_low, self.freq_bins_high, device=x.device
        ).view(-1, 1) # Forma: [num_freq_bins, 1]

        # 3. Calcular las ventanas Gaussianas en el dominio de la frecuencia
        # La ventana para la frecuencia 'n' es: G(m, n) = exp(-2 * pi^2 * m^2 / n^2)
        # Usamos una implementación estable que evita la división por cero si n=0 (ya lo evitamos con freq_bins_low>=1)
        # Forma de `windows`: [num_freq_bins, T]
        windows = torch.exp(-2 * (np.pi**2) * (freqs**2) / (target_freqs**2))
        
        # Expandir para broadcasting: [1, 1, num_freq_bins, T]
        windows = windows.unsqueeze(0).unsqueeze(0)

        # 4. Aplicar las ventanas al espectro "rolado" (shift cíclico)
        # Esto implementa la multiplicación H[m+n] * G[m] de la definición de la S-transform.
        
        # Crear una matriz para almacenar los espectros rolados
        # Forma: [B, C, num_freq_bins, T]
        rolled_ffts = torch.empty(
            (B, C, self.num_freq_bins, T), dtype=torch.complex64, device=x.device
        )
        
        # Rolar el espectro para cada frecuencia objetivo
        for i, n in enumerate(range(self.freq_bins_low, self.freq_bins_high)):
            rolled_ffts[:, :, i, :] = torch.roll(signal_fft, shifts=-n, dims=-1)
            
        # 5. Multiplicar los espectros rolados por las ventanas
        # [B, C, num_freq_bins, T] * [1, 1, num_freq_bins, T]
        windowed_ffts = rolled_ffts * windows

        # 6. Calcular la IFFT para obtener la S-transform
        # Forma: [B, C, num_freq_bins, T] (complejo)
        s_transform = torch.fft.ifft(windowed_ffts, dim=-1)
        
        return s_transform