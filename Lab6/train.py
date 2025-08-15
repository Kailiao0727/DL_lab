# train.py
import os
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# --- project-relative imports ---
from data.dataloader import ICLEVRDataset_train, ICLEVRDataset_eval
from torchvision import transforms


def build_train_transform(image_size: int = 64, augment: bool = False):
    ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)]
    if augment:
        ops += [transforms.RandomHorizontalFlip(p=0.5)]
    ops += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    # inverse of Normalize((0.5,)*3, (0.5,)*3)
    return x * 0.5 + 0.5




def sample_batch(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get one batch from a DataLoader."""
    it = iter(loader)
    batch = next(it)
    # train loader returns (imgs, labels); eval loader returns (labels, idx)
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        a, b = batch
        # try to figure out which is image by shape
        if torch.is_tensor(a) and a.ndim == 4:   # (B,3,H,W)
            imgs, labels = a, b
        else:
            imgs, labels = None, a  # label-only case
    else:
        raise RuntimeError("Unexpected batch structure")
    return imgs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--images_dirname", type=str, default="iclevr")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, args.images_dirname)

    # -------- Train loader (images + labels) --------
    train_tf = build_train_transform(image_size=args.image_size, augment=False)
    ds_train = ICLEVRDataset_train(
        transform=train_tf,
        data_path=data_dir,
        images_dir=args.images_dirname,   # your class joins with data_path
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    ds_test = ICLEVRDataset_eval(data_path=data_dir, split="test")
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )



if __name__ == "__main__":
    main()
