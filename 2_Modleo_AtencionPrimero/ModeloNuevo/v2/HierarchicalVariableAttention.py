# hier_vtt.py
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
import torch.nn.functional as F

class Attention(nn.Module):
    """ Módulo de atención genérico (self-attention o cross-attention) """
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, use_attn=False):
        h = self.heads
        # Si no se provee contexto, es self-attention
        context = x if context is None else context

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Estabilización numérica
        sim = torch.clamp(sim, min=-50.0, max=50.0)
        
        attn = sim.softmax(dim=-1)
        
        weights = attn if use_attn else None # Para visualización

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), weights

class HierarchicalVariableAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=16,
        dropout=0.,
        pooling_factor=2, # Factor para reducir instancias en el nivel grueso
        num_levels=2,     # Preparado para futuro, por ahora usamos 2
        mil=True          # Confirmamos que es para MIL
    ):
        super().__init__()
        assert num_levels == 2, "Actualmente solo se implementan 2 niveles (grueso/fino)."
        
        self.pooling_factor = pooling_factor
        self.dim = dim
        self.mil = mil

        # --- Nivel Grueso ---
        # Pooling para crear la vista de bajo detalle
        self.coarse_pool = nn.AvgPool2d(kernel_size=(pooling_factor, 1), stride=(pooling_factor, 1))
        # Auto-atención para el nivel grueso
        self.coarse_attn = Attention(dim, heads, dim_head, dropout)
        self.norm_coarse = nn.LayerNorm(dim)

        # --- Nivel Fino ---
        # Atención cruzada donde el nivel fino (query) atiende al nivel grueso (context)
        self.fine_cross_attn = Attention(dim, heads, dim_head, dropout)
        self.norm_fine = nn.LayerNorm(dim)
        
        # Capa FeedForward para refinar la salida
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(dim)


    def forward(self, x, use_attn=False, hierarchical=True):
        """
        x: Tensor de entrada con forma [B, T, F, D]
           B: batch_size, T: número de segmentos (bags), F: número de instancias, D: dimensión
        """
        if not hierarchical:
            # Si se desactiva la jerarquía, se comporta como la atención original (flat)
            # (Podríamos implementar una sola atención aquí si fuera necesario)
            print("Advertencia: HierarchicalVariableAttention llamada con hierarchical=False. Se devuelve la entrada.")
            return x, x 
            
        b, t, f, d = x.shape
        
        # Reshape para procesamiento: [B*T, F, D]
        x_flat = rearrange(x, 'b t f d -> (b t) f d')

        # --- Ruta Gruesa (Coarse Path) ---
        # 1. Pooling para reducir el número de instancias F
        # Para usar AvgPool2d, necesitamos una dimensión de canal: [(B*T), 1, F, D]
        x_pooled_input = rearrange(x_flat, 'bt f d -> bt 1 f d')
        x_pooled = self.coarse_pool(x_pooled_input)
        x_coarse_in = rearrange(x_pooled, 'bt 1 f d -> bt f d')

        # 2. Auto-atención en el nivel grueso
        coarse_attended, coarse_weights = self.coarse_attn(x_coarse_in, use_attn=use_attn)
        x_coarse_out = self.norm_coarse(coarse_attended + x_coarse_in)
        
        # --- Ruta Fina (Fine Path con guía Gruesa) ---
        # 1. Atención cruzada: Nivel Fino (q=x_flat) atiende a Nivel Grueso (k,v=x_coarse_out)
        fine_crossed, fine_weights = self.fine_cross_attn(x_flat, context=x_coarse_out, use_attn=use_attn)
        x_fine_intermediate = self.norm_fine(fine_crossed + x_flat)

        # 2. FFN para refinar
        x_fine_out = self.norm_ffn(self.ffn(x_fine_intermediate) + x_fine_intermediate)

        # --- Reconstruir la salida a la forma original ---
        f_coarse = x_coarse_out.shape[1]
        x_tok_coarse = rearrange(x_coarse_out, '(b t) f d -> b t f d', t=t, f=f_coarse)
        x_tok_fine = rearrange(x_fine_out, '(b t) f d -> b t f d', t=t)

        # Para MIL, devolvemos los embeddings de instancias para ambos niveles
        return x_tok_coarse, x_tok_fine