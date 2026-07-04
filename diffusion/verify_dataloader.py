import torch
import matplotlib.pyplot as plt
import os

from dataset import get_dataloader, BTEMP_VARIABLES

NC_ROOT_DIR = "/Users/dususu/Desktop/seaice_data/ready-to-train_train"
JSON_PATH = "/Users/dususu/Desktop/diffusion_training_dataset/dataset_split.json"

IMAGE_SIZE = 80
BATCH_SIZE = 4
NUM_SAMPLES_TO_VIZ = BATCH_SIZE
OUTPUT_DIR = "dataloader_verification"

print("--- Creating DataLoader for Verification ---")
verification_dataloader = get_dataloader(
    nc_root=NC_ROOT_DIR,
    json_path=JSON_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

def visualize_batch(batch, num_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    damaged_batch = batch["damaged"]
    clean_batch = batch["clean"]
    file_ids = batch["id"]

    damaged_batch = (damaged_batch + 1) / 2
    clean_batch = (clean_batch + 1) / 2

    for i in range(min(num_samples, damaged_batch.shape[0])):
        num_btemp = len(BTEMP_VARIABLES)
        fig, axes = plt.subplots(num_btemp, 2, figsize=(10, 2 * num_btemp))

        file_id = file_ids[i]
        fig.suptitle(f"Verification for Sample ID: {file_id}", fontsize=16)

        for j, var_name in enumerate(BTEMP_VARIABLES):
            damaged_channel = damaged_batch[i, j, :, :]
            clean_channel = clean_batch[i, j, :, :]

            ax_damaged = axes[j, 0]
            ax_damaged.imshow(damaged_channel.cpu().numpy(), cmap='jet')
            ax_damaged.set_title(f"Input (Damaged)\n{var_name}")
            ax_damaged.axis('off')

            ax_clean = axes[j, 1]
            ax_clean.imshow(clean_channel.cpu().numpy(), cmap='jet')
            ax_clean.set_title(f"Target (Clean)\n{var_name}")
            ax_clean.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(output_dir, f"verification_sample_{i + 1}_{file_id}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"  -> Saved verification image to: {save_path}")

if __name__ == "__main__":
    print("\n--- Fetching a Batch from DataLoader ---")
    try:
        sample_batch = next(iter(verification_dataloader))

        if sample_batch:
            print("Successfully fetched a batch. Starting visualization...")
            visualize_batch(sample_batch, NUM_SAMPLES_TO_VIZ, OUTPUT_DIR)
            print("\n--- Verification Finished! ---")
            print(f"Please check the '{OUTPUT_DIR}' folder for results.")
        else:
            print("DataLoader returned an empty batch. Cannot verify.")

    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")