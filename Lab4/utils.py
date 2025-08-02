import matplotlib.pyplot as plt
import numpy as np

# Suppose avg_psnr_per_frame is from the above step
def plot_psnr(avg_psnr_per_frame):
    frames = np.arange(1, len(avg_psnr_per_frame)+1)
    plt.figure(figsize=(10,5))
    plt.plot(frames, avg_psnr_per_frame, marker='o')
    plt.title("PSNR per Frame in Validation Set")
    plt.xlabel("Frame Index (t)")
    plt.ylabel("PSNR (dB)")
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example usage:
# avg_psnr_per_frame = model.eval()
# plot_psnr(avg_psnr_per_frame)
