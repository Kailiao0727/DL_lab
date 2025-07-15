import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from matplotlib.colors import ListedColormap

def dice_score(pred, ground_truth, epsilon=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * ground_truth).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + ground_truth.sum(dim=(1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def plot(train_losses, epochs, show=True):
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputs/training_loss_curve.png")

    if(show==True):
        plt.show()

def visualize(image, prediction, mask, save=True):
    image = image.cpu()
    mask = mask.cpu()
    prediction = (prediction > 0.5).float().cpu()

    custom_cmap = ListedColormap(["purple", "yellow"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(to_pil_image(image))
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(mask[0], cmap=custom_cmap)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(prediction[0], cmap=custom_cmap)
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    if(save):
        plt.savefig("outputs/visualization.png")
    