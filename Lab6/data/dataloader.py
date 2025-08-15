import json
import os
from typing import List, Dict, Tuple


import torch
from torch.utils.data import Dataset
from PIL import Image

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _to_multi_hot(names: List[str], object2idx: Dict[str, int]) -> torch.Tensor:
    v = torch.zeros(len(object2idx), dtype=torch.float32)
    for n in names:
        v[object2idx[n]] = 1.0
    return v

class ICLEVRDataset_train(Dataset):
    def __init__(self, transform, data_path, images_dir = 'iclevr'):
        super().__init__()
        self.transform = transform
        self.data_path = data_path
        self.image_path = os.path.join(data_path, images_dir)
        self.object2idx: Dict[str, int] = load_json(os.path.join(data_path, 'objects.json'))
        self.mapping: Dict[str, List[str]] = load_json(os.path.join(data_path, 'train.json'))
        
        
        self.items: List[Tuple[str, torch.Tensor]] = []
        for fname, names in self.mapping.items():
            self.items.append((fname, _to_multi_hot(names, self.object2idx)))


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label_vec = self.items[idx]
        path = os.path.join(self.image_path, fname)
        
        with Image.open(path).convert("RGB") as im:
            x = self.transform(im)
        
        return x, label_vec
    

class ICLEVRDataset_eval(Dataset):
    def __init__(self, data_path, split='test'):
        super().__init__()
        self.data_path = data_path
        self.object2idx: Dict[str, int] = load_json(os.path.join(data_path, 'objects.json'))
        self.label_sets: List[List[str]] = load_json(os.path.join(data_path, f'{split}.json'))
        
        
        self.items: List[torch.Tensor] = []
        for i in self.label_sets:
            self.items.append( _to_multi_hot(i, self.object2idx))


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx], idx