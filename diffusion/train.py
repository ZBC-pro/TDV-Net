import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from accelerate import Accelerator
from accelerate.utils import set_seed

try:
    from dataset_enhance import get_dataloader, CONTEXT_VARIABLES
except ImportError:
    pass

NC_ROOT_DIR = 'ready-to-train_train'
JSON_PATH = 'AMSR2_diffusion/Data extraction and cleaning/dataset_split.json'
OUTPUT_DIR = f"./diff_model"

IMAGE_SIZE = 256
BATCH_SIZE = 8 
LEARNING_RATE = 1e-5
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 10
VISUALIZATION_DIR = f"./noise_visualization"

def visualize_noise_steps(dataloader, scheduler, device, output_dir):
    print("\n--- Starting noise visualization step (on already damaged data)... ---")
    timesteps_to_visualize = [0, 150, 300, 450, 700, 850, 999]

    try:
        vis_batch = next(iter(dataloader))
        if vis_batch is None: return
    except StopIteration:
        return

    damaged_input = vis_batch["damaged_input"].to(device)
    target_channel_with_seamline = damaged_input[:, 0:1, :, :]
    num_samples_to_vis = min(10, target_channel_with_seamline.shape[0])

    for t_step in timesteps_to_visualize:
        timesteps = torch.tensor([t_step] * num_samples_to_vis, device=device)
        noise = torch.randn_like(target_channel_with_seamline[:num_samples_to_vis])
        noisy_images = scheduler.add_noise(target_channel_with_seamline[:num_samples_to_vis], noise, timesteps)
        noisy_images_denorm = noisy_images.clamp(-1, 1) * 0.5 + 0.5

        for i in range(num_samples_to_vis):
            img_array = noisy_images_denorm[i, 0].cpu().numpy()
            output_path = os.path.join(output_dir, f"damaged_{i:02d}_{t_step}.png")
            plt.imsave(output_path, img_array, cmap='viridis', vmin=0, vmax=1)

    print(f"--- Visualization finished. Images saved to '{output_dir}'. ---")

def main():
    accelerator = Accelerator(mixed_precision="fp16") 
    set_seed(42)

    if accelerator.is_main_process:
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if accelerator.is_main_process:
        print("Creating DataLoaders...")
        
    train_dataloader = get_dataloader(
        nc_root=NC_ROOT_DIR,
        json_path=JSON_PATH,
        split='train',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        use_augmentation=True
    )

    validation_dataloader = get_dataloader(
        nc_root=NC_ROOT_DIR,
        json_path=JSON_PATH,
        split='validation',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        use_augmentation=False
    )

    if accelerator.is_main_process:
        print(f"Setting up the generic 1+{len(CONTEXT_VARIABLES)}-channel model...")
        
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=1 + len(CONTEXT_VARIABLES),
        out_channels=1, 
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * max_train_steps),
        num_training_steps=max_train_steps
    )

    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )

    device = accelerator.device 

    if accelerator.is_main_process:
        print(f"\n--- Starting Model Training on {accelerator.num_processes} GPUs ---")

    best_validation_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        if epoch == 0 and accelerator.is_main_process:
            visualize_noise_steps(
                dataloader=train_dataloader,
                scheduler=noise_scheduler,
                device=device,
                output_dir=VISUALIZATION_DIR
            )

        model.train()
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Training]", disable=not accelerator.is_main_process)

        for step, batch in enumerate(train_loop):
            if batch is None: continue

            damaged_input = batch["damaged_input"] 
            clean_target = batch["clean_target"]

            noise = torch.randn_like(clean_target)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_target.shape[0],), device=device)
            
            noisy_target = noise_scheduler.add_noise(clean_target, noise, timesteps)

            model_input = damaged_input.clone()
            model_input[:, 0:1, :, :] = noisy_target

            predicted_noise = model(model_input, timesteps).sample
            loss = F.mse_loss(predicted_noise, noise)

            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        
        total_validation_mse = 0
        total_validation_rmse = 0
        total_validation_mae = 0
        total_validation_r2 = 0
        num_validation_batches = 0
        
        validation_loop = tqdm(validation_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Validation]", disable=not accelerator.is_main_process)
        
        with torch.no_grad():
            for validation_batch in validation_loop:
                if validation_batch is None: continue
        
                damaged_input = validation_batch["damaged_input"]
                clean_target = validation_batch["clean_target"]
        
                noise = torch.randn_like(clean_target)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_target.shape[0],), device=device)
                noisy_target = noise_scheduler.add_noise(clean_target, noise, timesteps)
        
                model_input = damaged_input.clone()
                model_input[:, 0:1, :, :] = noisy_target
        
                predicted_noise = model(model_input, timesteps).sample

                all_predictions, all_targets = accelerator.gather_for_metrics((predicted_noise, noise))

                val_loss_mse = F.mse_loss(all_predictions, all_targets)
                val_loss_mae = F.l1_loss(all_predictions, all_targets)
                val_loss_rmse = torch.sqrt(val_loss_mse)

                target_mean = torch.mean(all_targets, dim=[1, 2, 3], keepdim=True)
                ss_tot = torch.sum((all_targets - target_mean) ** 2, dim=[1, 2, 3])
                ss_res = torch.sum((all_targets - all_predictions) ** 2, dim=[1, 2, 3])
                r2_scores = 1 - (ss_res / (ss_tot + 1e-8))
                val_r2 = torch.mean(r2_scores)

                total_validation_mse += val_loss_mse.item()
                total_validation_rmse += val_loss_rmse.item()
                total_validation_mae += val_loss_mae.item()
                total_validation_r2 += val_r2.item()
                num_validation_batches += 1
        
        if num_validation_batches > 0:
            avg_val_mse = total_validation_mse / num_validation_batches
            avg_val_rmse = total_validation_rmse / num_validation_batches
            avg_val_mae = total_validation_mae / num_validation_batches
            avg_val_r2 = total_validation_r2 / num_validation_batches
            
            if accelerator.is_main_process:
                print(f"--- Epoch {epoch + 1} | Avg Val MSE: {avg_val_mse:.6f} | R2: {avg_val_r2:.6f} ---")
            
                if avg_val_mse < best_validation_loss:
                    print(f"Validation MSE improved ({best_validation_loss:.4f} -> {avg_val_mse:.4f}). Saving model...")
                    best_validation_loss = avg_val_mse
                    best_model_output_dir = os.path.join(OUTPUT_DIR, "best_model")
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(best_model_output_dir)
                    patience_counter = 0
                else:
                    print(f"No improvement. Patience: {patience_counter+1}/{EARLY_STOPPING_PATIENCE}")
                    patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if accelerator.is_main_process:
                print("Early stopping triggered.")
            break

    if accelerator.is_main_process:
        print(f"\n--- Training Finished! Best MSE: {best_validation_loss:.4f} ---")

if __name__ == "__main__":
    main()