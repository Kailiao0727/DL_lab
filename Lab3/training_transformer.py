import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData, plot_loss_curves
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        self.train_losses = []
        self.val_losses = []
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, loader, epoch, args):
        self.model.train()
        total_loss = 0

        for i, x in enumerate(tqdm(loader)):
            x = x.to(args.device)

            logits, z_indices = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))

            loss.backward()

            if (i + 1) % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

            total_loss += loss.item()
            

        self.scheduler.step()
        avg_loss = total_loss / len(loader)
        self.train_losses.append(avg_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    def eval_one_epoch(self, loader, epoch, args):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, x in enumerate(tqdm(loader)):
                x = x.to(args.device)

                logits, z_indices = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))

                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            self.val_losses.append(avg_loss)

        print(f"[Epoch {epoch}] Valid Loss: {avg_loss:.4f}")

    def configure_optimizers(self, args):
        optimizer = torch.optim.AdamW(
        self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    
    
    torch.cuda.set_device(args.device)
    print("Running on:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    min_valid_loss = float('inf')
    if os.path.exists(args.checkpoint_path) and not args.restart:
        checkpoint = torch.load(args.checkpoint_path)
        train_transformer.model.load_state_dict(checkpoint['model_state'])
        train_transformer.optim.load_state_dict(checkpoint['optimizer_state'])
        train_transformer.scheduler.load_state_dict(checkpoint['scheduler_state'])
        train_transformer.train_losses = checkpoint['train_losses']
        train_transformer.val_losses = checkpoint['val_losses']
        args.start_from_epoch = checkpoint['epoch']
        min_valid_loss = min(checkpoint['val_losses'])
        if args.plot:
            plot_loss_curves(train_transformer.train_losses, train_transformer.val_losses)
            exit()
        print(f"Resumed from checkpoint: epoch {args.start_from_epoch}")
        print(f"Previous train loss: {train_transformer.train_losses[-1]}")

#TODO2 step1-5:
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_transformer.train_one_epoch(train_loader, epoch, args)
        train_transformer.eval_one_epoch(val_loader, epoch, args)

        if epoch % args.save_per_epoch == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': train_transformer.model.state_dict(),
                'optimizer_state': train_transformer.optim.state_dict(),
                'scheduler_state': train_transformer.scheduler.state_dict(),
                'train_losses': train_transformer.train_losses,
                'val_losses': train_transformer.val_losses
            }, args.checkpoint_path)
            print(f"Checkpoint saved to {args.checkpoint_path}")
            if (train_transformer.val_losses[-1] < min_valid_loss):
                torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best_val.pth")