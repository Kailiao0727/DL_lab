# train.py
import os
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# --- project-relative imports ---
from data.dataloader import ICLEVRDataset_train, ICLEVRDataset_eval
from models import ConditionalUNet, DDPMWrapper
from evaluator.evaluator import evaluation_model


def build_train_transform(image_size: int = 64, augment: bool = False):
    ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)]
    if augment:
        ops += [transforms.RandomHorizontalFlip(p=0.5)]
    ops += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    # inverse of Normalize((0.5,)*3, (0.5,)*3)
    return x * 0.5 + 0.5

def train(args, device):
    eval_model = evaluation_model()
    train_tf = build_train_transform(image_size=args.image_size, augment=False)
    ds_train = ICLEVRDataset_train(transform=train_tf, data_path=args.data_dir, images_dir=args.images_dirname)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    unet = ConditionalUNet(sample_size=args.image_size, label_dim=24, label_emb_size=512)
    ddpm = DDPMWrapper(unet, num_train_timesteps=args.num_timesteps).to(device)

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        ddpm.train()
        total_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")
        for x0, labels in pbar:
            x0, labels = x0.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = ddpm.loss_step(x0, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x0.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dl_train)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            ddpm.eval()
            with torch.no_grad():
                labels_eval = labels[:8]   
                samples = ddpm.sample(labels_eval, num_steps=args.num_timesteps, batch_size=8, device=device)

            grid = denormalize(samples.clamp(-1,1))
            os.makedirs("outputs", exist_ok=True)
            save_image(grid, f"outputs/samples_epoch_{epoch+1}.png", nrow=8)
            
            acc = eval_model.eval(samples, labels_eval)
            print(f"Epoch {epoch+1} - Evaluation Accuracy: {acc:.4f}")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "unet":unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            }, f"checkpoints/unet_{epoch+1}.pth")

        
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--images_dirname", type=str, default="iclevr")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(args, device)



if __name__ == "__main__":
    main()
