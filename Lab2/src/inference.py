import argparse
import torch
import os

from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34_UNet
from src.oxford_pet import load_dataset
from src.utils import dice_score, visualize
from torch.utils.data import DataLoader


def inference(model_name, data_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    if model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    if model_name == "resnet34_unet":
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)

    model_path = os.path.join("saved_models", model_name + ".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = load_dataset(data_path, mode="test")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    for batch in test_loader:
        print("Image shape:", batch["image"].shape)
        print("Mask shape:", batch["mask"].shape)
        print("batches: ", len(test_loader))
        break

    dices = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)
            batch_dice = dice_score(outputs, masks)
            dices.append(batch_dice)

            if idx == 1:
                visualize(images, outputs, masks, save=True)
        avg_dice = sum(dices) / len(dices)
        print(f"Average Dice score: {avg_dice:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='unet', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, default='dataset', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args.model, args.data_path, args.batch_size)