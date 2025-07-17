import argparse
import torch
import os

from src.models.unet import UNet
from src.oxford_pet import load_dataset
from src.utils import dice_score, visualize
from torch.utils.data import DataLoader

def evaluate(model, val_loader, device):  
    model.eval()
                            
    criterion = torch.nn.BCELoss()
    total_loss = 0

    dices = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)
            dices.append(dice_score(outputs, masks))
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    avg_dice = sum(dices) / len(dices)
    return avg_loss, avg_dice


    