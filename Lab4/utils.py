from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

# === SETTINGS ===
log_path = "logs_mono_tfr0.8/events.out.tfevents.1753975618.gpuserv4.240378.0"
target_epoch = 96  # Change this to the epoch you're interested in

# === Load event log ===
ea = EventAccumulator(log_path)
ea.Reload()

# === Prepare storage ===
frame_psnr = []

print("Available Scalar Tags:")
for tag in ea.Tags()['scalars']:
    print(tag)
# Loop over all scalar tags
for tag in ea.Tags()['scalars']:
    if tag.startswith("PSNR/frame_"):
        entries = ea.Scalars(tag)
        # Find the entry with matching epoch
        for e in entries:
            if e.step == target_epoch:
                frame_index = int(tag.split("_")[-1])
                frame_psnr.append((frame_index, e.value))
                break  # Stop once we find the matching step

# Sort by frame index
frame_psnr.sort(key=lambda x: x[0])

# Save as CSV
df = pd.DataFrame(frame_psnr, columns=["frame_index", f"psnr_at_epoch_{target_epoch}"])
df.to_csv(f"psnr_epoch_{target_epoch}.csv", index=False)

print(f"✅ Saved PSNR values of all frames at epoch {target_epoch} to CSV.")

