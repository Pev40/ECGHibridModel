import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot_snomed_bar(snomed_embed, coarse_names=None, normalize=True, sort=True, figsize=(6, 4), color='tab:red', title='Embedding SNOMED (coarse)'):
    """
    Grafica un vector snomed_embed como barras horizontales.

    snomed_embed: array-like [num_coarse] o [1, num_coarse]
    coarse_names: lista de nombres de clases coarse (len == num_coarse)
    """
    vals = _to_numpy(snomed_embed).squeeze()
    if vals.ndim != 1:
        raise ValueError('snomed_embed debe ser un vector 1D o [1, C].')
    if normalize and vals.sum() > 0:
        vals = vals / vals.sum()
    labels = np.array(coarse_names) if coarse_names is not None else np.array([f'C{i}' for i in range(len(vals))])
    order = np.argsort(vals) if sort else np.arange(len(vals))
    plt.figure(figsize=figsize)
    plt.barh(labels[order], vals[order], color=color)
    plt.xlabel('Peso')
    plt.title(title)
    plt.tight_layout()
    return plt.gca()


def plot_snomed_heatmap(snomed_batch, coarse_names=None, figsize=(8, 4), cmap='Reds', title='Heatmap SNOMED (muestras × coarse)'):
    """
    Grafica un heatmap para múltiples snomed_embed apilados.

    snomed_batch: array-like [N, num_coarse]
    """
    M = _to_numpy(snomed_batch)
    if M.ndim != 2:
        raise ValueError('snomed_batch debe ser de forma [N, C].')
    plt.figure(figsize=figsize)
    ax = sns.heatmap(M, cmap=cmap, cbar=True, xticklabels=(coarse_names if coarse_names is not None else True), yticklabels=False)
    plt.title(title)
    plt.xlabel('Grupos coarse')
    plt.ylabel('Muestras')
    plt.tight_layout()
    return ax


def plot_snomed_projection(snomed_batch, model=None, device=None, n_components=2, figsize=(5, 4), title='Proyección 2D de snomed_proj(snomed_embed)'):
    """
    Proyecta snomed_embed con la capa model.snomed_proj (si se provee) y aplica PCA a 2D para graficar.

    snomed_batch: array-like [N, num_coarse]
    model: modelo HMST con atributo snomed_proj (opcional). Si None, proyecta identidad.
    device: torch.device para ejecutar la proyección del modelo.
    """
    from sklearn.decomposition import PCA

    M = snomed_batch
    if isinstance(M, np.ndarray):
        M_t = torch.from_numpy(M).float()
    elif isinstance(M, torch.Tensor):
        M_t = M.float().detach().cpu()
    else:
        M_t = torch.tensor(M).float()

    if model is not None and hasattr(model, 'snomed_proj'):
        dev = device or next(model.parameters()).device
        with torch.no_grad():
            Z = model.snomed_proj(M_t.to(dev)).cpu().numpy()
    else:
        Z = M_t.numpy()

    if Z.shape[1] < n_components:
        # rellena con ceros si la dim es menor
        pad = n_components - Z.shape[1]
        Z = np.pad(Z, ((0, 0), (0, pad)), mode='constant')

    Z2 = PCA(n_components=n_components).fit_transform(Z)
    plt.figure(figsize=figsize)
    plt.scatter(Z2[:, 0], Z2[:, 1], s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    return plt.gca()


def collect_snomed_from_loader(loader, max_batches=None, prefer_item_key='snomed_embed', fallback_key='labels_coarse'):
    """
    Extrae una matriz [N, C] de embeddings SNOMED desde un DataLoader/Dataset.
    - Busca primero el key prefer_item_key (e.g., 'snomed_embed'), si no existe usa fallback_key (e.g., 'labels_coarse').
    - Concatena a lo largo de las muestras hasta max_batches si se pasa.
    """
    xs = []
    seen = 0
    for bi, batch in enumerate(loader):
        item = batch.get(prefer_item_key)
        if item is None:
            item = batch.get(fallback_key)
        if item is None:
            raise KeyError(f"Ni '{prefer_item_key}' ni '{fallback_key}' están presentes en el batch.")
        xs.append(_to_numpy(item))
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    if len(xs) == 0:
        raise ValueError('Loader no produjo elementos con las claves esperadas.')
    M = np.concatenate(xs, axis=0)
    return M


def plot_snomed_from_loader_bar(loader, coarse_names, idx=0, prefer_item_key='snomed_embed', fallback_key='labels_coarse', **kwargs):
    """
    Toma un DataLoader, extrae el primer batch con el vector coarse y dibuja barra del elemento idx.
    """
    for batch in loader:
        item = batch.get(prefer_item_key)
        if item is None:
            item = batch.get(fallback_key)
        if item is None:
            raise KeyError(f"Ni '{prefer_item_key}' ni '{fallback_key}' están presentes en el batch.")
        vec = item[idx]
        return plot_snomed_bar(vec, coarse_names=coarse_names, **kwargs)
    raise RuntimeError('Loader vacío.')


def plot_snomed_from_loader_heatmap(loader, coarse_names=None, max_batches=10, prefer_item_key='snomed_embed', fallback_key='labels_coarse', **kwargs):
    """
    Construye heatmap acumulando hasta max_batches del loader.
    """
    M = collect_snomed_from_loader(loader, max_batches=max_batches, prefer_item_key=prefer_item_key, fallback_key=fallback_key)
    return plot_snomed_heatmap(M, coarse_names=coarse_names, **kwargs)


def plot_snomed_from_loader_projection(loader, model=None, device=None, max_batches=10, prefer_item_key='snomed_embed', fallback_key='labels_coarse', **kwargs):
    """
    Proyección 2D usando datos reales del loader.
    """
    M = collect_snomed_from_loader(loader, max_batches=max_batches, prefer_item_key=prefer_item_key, fallback_key=fallback_key)
    return plot_snomed_projection(M, model=model, device=device, **kwargs)


