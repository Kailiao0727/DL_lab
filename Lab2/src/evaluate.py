import argparse
import torch
import os

from src.models.unet import UNet
from src.oxford_pet import load_dataset
from src.utils import dice_score, visualize
from torch.utils.data import DataLoader



def evaluate(model_name, data_path, batch_size=16):
    # implement the evaluation function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    if model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)

    model_path = os.path.join("saved_models", model_name + ".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    valid_dataset = load_dataset(data_path, mode="valid")

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)

    for batch in valid_loader:
        print("Image shape:", batch["image"].shape)
        print("Mask shape:", batch["mask"].shape)
        print("batches: ", len(valid_loader))
        break

    dices = []
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)
            batch_dice = dice_score(outputs, masks)
            dices.append(batch_dice)

            if idx == 1:
                visualize(images[0], outputs[0], masks[0], save=True)
        avg_dice = sum(dices) / len(dices)
        print(f"Average Dice score: {avg_dice:.4f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--data_path', type=str, default='dataset', help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    evaluate(model_name=args.model_name, data_path=args.data_path, batch_size=args.batch_size)
    