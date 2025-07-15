import argparse
import os
import torch

from src.oxford_pet import load_dataset
from torch.utils.data import DataLoader
from src.models.unet import UNet
from src.utils import plot
from tqdm import tqdm



def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = load_dataset(args.data_path, mode="train")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    for batch in train_loader:
        print("Image shape:", batch["image"].shape)
        print("Mask shape:", batch["mask"].shape)
        print("batches: ", len(train_loader))
        break

    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint_path = os.path.join("saved_models", args.model + ".pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()

    train_losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, "saved_models/unet_checkpoint.pth")


    torch.save(model.state_dict(), "saved_models/"+args.model+".pth")
    plot(train_losses=train_losses, epochs=args.epochs, show=True)
    print("Model weights saved to saved_models/unet.pth")
    



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', type=str, default='unet', help='name of the model to save')
    parser.add_argument('--data_path', type=str, default='dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
    # plot(train_losses=[0.5387, 0.3901, 0.3494, 0.3217, 0.2998, 0.2821, 0.2675, 0.2545, 0.2395, 0.2266, 0.2142, 0.2008, 0.1885, 0.1779, 0.1631,
    # 0.1564, 0.1436, 0.1370, 0.1299, 0.1227, 0.1196, 0.1134, 0.1066, 0.1002, 0.0984], epochs=25, show=True)