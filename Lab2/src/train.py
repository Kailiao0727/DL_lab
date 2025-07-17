import argparse
import os
import torch

from src.oxford_pet import load_dataset
from torch.utils.data import DataLoader
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34_UNet
from src.utils import plot, DSCPlusPlusLoss
from src.evaluate import evaluate
from tqdm import tqdm



def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)

    for batch in train_loader:
        print("Image shape:", batch["image"].shape)
        print("Mask shape:", batch["mask"].shape)
        print("batches: ", len(train_loader))
        break

    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.model == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    checkpoint_path = os.path.join("saved_models", args.model + "_checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        prev_loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}, previous loss: {prev_loss:.4f}")
    else:
        start_epoch = 0

    criterion = torch.nn.BCELoss()
    # criterion = DSCPlusPlusLoss(gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    dice_loss_fn = DSCPlusPlusLoss(gamma=0.5)
    bce_loss_fn = torch.nn.BCELoss()
    train_losses = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0

        # alpha = epoch / args.epochs
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)
            loss_dice = dice_loss_fn(outputs, masks)
            loss_bce = bce_loss_fn(outputs, masks)
            loss = (1 - 0.5) * loss_dice + 0.5 * loss_bce
            # loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_loss, val_dice = evaluate(model, valid_loader, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_loss:.4f}, Val Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, "saved_models/"+ args.model + "_checkpoint.pth")
        torch.save(model.state_dict(), "saved_models/"+args.model+".pth")
    plot(train_losses=train_losses, epochs=args.epochs, model=args.model, show=True)
    print("Model weights saved to saved_models/unet.pth")
    



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', type=str, default='unet', help='name of the model to save')
    parser.add_argument('--data_path', type=str, default='dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume training from last checkpoint')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
    # plot(train_losses=[0.5387, 0.3901, 0.3494, 0.3217, 0.2998, 0.2821, 0.2675, 0.2545, 0.2395, 0.2266, 0.2142, 0.2008, 0.1885, 0.1779, 0.1631,
    # 0.1564, 0.1436, 0.1370, 0.1299, 0.1227, 0.1196, 0.1134, 0.1066, 0.1002, 0.0984, 0.0939, 0.0895, 0.0845, 0.0827, 0.0812, 0.0732, 0.0736, 0.0788, 0.0656, 0.0680], epochs=25, show=True)