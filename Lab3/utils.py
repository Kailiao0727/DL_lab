from torch.utils.data import Dataset as torchData
from glob import glob
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
import os
import torch.nn as nn
import matplotlib.pyplot as plt

class LoadTrainData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), #Convert to tensor
                transforms.Normalize(mean=[0.4816, 0.4324, 0.3845],std=[0.2602, 0.2518, 0.2537]),# Normalize the pixel values
        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        #self.folder = glob(os.path.join(root + '/*.png'))
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Training Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadTestData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.4868, 0.4341, 0.3844],std=[0.2620, 0.2527, 0.2543]),
                        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Testing Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadMaskData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Mask Data For Inpainting Task: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))


def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    