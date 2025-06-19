import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np
import multiprocessing



class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        # Load samples
        x_data = dataset["samples"]

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None

        self.len = x_data.shape[0]

    def get_labels(self):
        return self.y_data

    def __getitem__(self, index):
        sample = {
            'samples': self.x_data[index].squeeze(-1),
            'labels': int(self.y_data[index])
        }

        return sample

    def __len__(self):
        return self.len


def data_generator(data_path, data_type, hparams):
    # original
    train_dataset = torch.load(os.path.join(data_path, data_type, f"train.pt"))
    val_dataset = torch.load(os.path.join(data_path, data_type, f"val.pt"))
    test_dataset = torch.load(os.path.join(data_path, data_type, f"test.pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    cw = train_dataset.y_data.numpy().tolist()
    cw_dict = {}
    for i in range(len(np.unique(train_dataset.y_data.numpy()))):
        cw_dict[i] = cw.count(i)
    # print(cw_dict)

    # Configuración optimizada para eficiencia de GPU
    batch_size = hparams["batch_size"]
    
    # Determinar número óptimo de workers
    num_workers = min(4, multiprocessing.cpu_count())  # Máximo 4 workers para evitar overhead
    
    # Configuración optimizada de DataLoader
    dataloader_config = {
        'batch_size': batch_size,
        'pin_memory': True,  # Acelera transferencia a GPU
        'num_workers': num_workers,
        'persistent_workers': True if num_workers > 0 else False,  # Mantiene workers vivos
        'prefetch_factor': 2 if num_workers > 0 else None,  # Pre-carga 2 batches por worker
        'drop_last': True
    }
    
    # Dataloaders optimizados
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        shuffle=True, 
        **dataloader_config
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        shuffle=False, 
        **{**dataloader_config, 'drop_last': True}
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        shuffle=False, 
        **{**dataloader_config, 'drop_last': False}
    )
    
    print(f"📊 DataLoader optimizado configurado:")
    print(f"   - Workers: {num_workers}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Pin memory: True")
    print(f"   - Persistent workers: {dataloader_config['persistent_workers']}")
    
    return train_loader, val_loader, test_loader, get_class_weight(cw_dict)


import math


def get_class_weight(labels_dict):
    total = sum(labels_dict.values())
    max_num = max(labels_dict.values())
    mu = 1.0 / (total / max_num)
    class_weight = dict()
    for key, value in labels_dict.items():
        score = math.log(mu * total / float(value))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight
