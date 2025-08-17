import os
import argparse
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from evaluator.evaluator import evaluation_model
from data.dataloader import ICLEVRDataset_eval
from models import ConditionalUNet, DDPMWrapper


def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1] for visualization
    return x.clamp(-1, 1) * 0.5 + 0.5

def get_score(ddpm, dl, device, eval_model, split_name):
    ddpm.eval()

    total_acc = 0.0
    total_batches = 0
    rows_for_grid = []

    img_index = 0
    os.makedirs(f'outputs/{split_name}', exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Evaluating {split_name} dataset"):
            labels, _ = batch  # labels are the first element
            labels = labels.to(device)

            # Sample images from the DDPM model
            sampled_images = ddpm.sample(labels=labels, device=device)

            # Evaluate the sampled images
            acc = eval_model.eval(sampled_images, labels)
            total_acc += acc
            total_batches += 1

            for i in range(sampled_images.size(0)):
                save_path = f'outputs/{split_name}/{img_index:03d}.png'
                save_image(denorm(sampled_images[i].cpu()), save_path)
                img_index += 1

            row = make_grid(denorm(sampled_images.detach().cpu()), nrow=sampled_images.size(0))
            rows_for_grid.append(row)

    full_grid = torch.cat(rows_for_grid, dim=1)
    grid_path = f'outputs/{split_name}_grid.png'
    save_image(full_grid, grid_path)

    avg_acc = total_acc / total_batches
    print(f"Average accuracy on {split_name}: {avg_acc:.4f}")
    print(f"Saved grid: {grid_path}")

    return avg_acc

def save_denoising_process(ddpm, labels, device, num_steps=1000):
    ddpm.eval()
    labels = labels.to(device)

    B = labels.size(0)
    x = torch.randn(B, 3, 64, 64, device=device)

    snapshots = []
    step_interval = num_steps // 10

    for t in reversed(range(num_steps)):
        x = ddpm.p_sample(x, t, labels)
        if t % step_interval == 0:
            snapshots.append(denorm(x.detach().cpu()))
    
    imgs = [snap[0:1] for snap in snapshots]
    grid = make_grid(torch.cat(imgs, dim=0), nrow=len(imgs))
    out_path = 'outputs/denoising_process.png'
    save_image(grid, out_path)
    print(f"Denoising process saved to {out_path}")

def main(args):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data_dir
    batch_size = 8

    ds_new_test = ICLEVRDataset_eval(data_path=data_dir, split="new_test")
    dl_new_test = torch.utils.data.DataLoader(
        ds_new_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    ds_test = ICLEVRDataset_eval(data_path=data_dir, split="test")
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    unet = ConditionalUNet().to(device)
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    unet.load_state_dict(checkpoint['unet'])
    ddpm = DDPMWrapper(unet).to(device)

    eval_model = evaluation_model()

    for i, batch in enumerate(dl_test):
        if i == 3:
            lbl, _ = batch
            print("label map for denoising process: ", lbl[1])
            save_denoising_process(ddpm, labels=lbl[1].unsqueeze(0), device=device)
            break

    acc_test = get_score(ddpm, dl_test, device, eval_model, "test")
    acc_new_test = get_score(ddpm, dl_new_test, device, eval_model, "new_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/unet_100.pth")
    args = parser.parse_args()
    main(args)
    


