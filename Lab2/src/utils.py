import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn

class DSCPlusPlusLoss(nn.Module):
    def __init__(self, gamma=0.5, smooth=1e-5):
        """
        gamma: Controls strength of penalty for overconfident errors (default 0.5 per paper)
        smooth: Small value to avoid divide-by-zero
        """
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (N, 1, H, W) - predicted probability maps (after sigmoid)
        # targets: (N, 1, H, W) - ground truth masks (0/1)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Modulating factor (focal-style)
        mod_factor = torch.abs(inputs - targets) ** self.gamma

        intersection = torch.sum(mod_factor * inputs * targets, dim=1)
        denominator = torch.sum(mod_factor * inputs, dim=1) + torch.sum(mod_factor * targets, dim=1)
        dice_score = (2 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1 - dice_score
        return loss.mean()

def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    loss = 1 - dice
    return loss.mean()
    

def dice_score(pred, ground_truth, epsilon=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * ground_truth).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + ground_truth.sum(dim=(1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def plot(train_losses, epochs, model, show=True):
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputs/"+ model + ".png")

    if(show==True):
        plt.show()

def visualize(images, predictions, masks, save=True):
    images = images.cpu()
    masks = masks.cpu()
    predictions = (predictions > 0.5).float().cpu()

    batch_size = min(images.size(0), 4)
    print(images.shape)
    custom_cmap = ListedColormap(["purple", "yellow"])

    fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    if batch_size == 1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(batch_size):
        axs[i, 0].imshow(to_pil_image(images[i]))
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(masks[i][0], cmap=custom_cmap)
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(predictions[i][0], cmap=custom_cmap)
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
    if(save):
        plt.savefig("outputs/visualization.png")
    