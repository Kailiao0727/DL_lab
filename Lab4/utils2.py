import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file assuming it's been generated and uploaded
csv_path = "psnr_epoch_117.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df["frame_index"], df[df.columns[1]])
plt.xlabel("Frame Index")
plt.ylabel("PSNR")
plt.title(f"Per-frame PSNR at Epoch 100")
plt.grid(True)
plt.tight_layout()
plt.show()