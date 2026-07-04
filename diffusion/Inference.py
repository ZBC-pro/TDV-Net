import torch
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from tqdm import tqdm
import traceback
import random

try:
    from dataset_enhance import CONTEXT_VARIABLES
except ImportError:
    CONTEXT_VARIABLES = ['skt', 't2m', 'tcwv', 'u10m_rotated', 'v10m_rotated']

IMAGE_SIZE = 256
TRAIN_TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_array_as_pure_png(data_array, output_path, cmap='viridis', vmin=None, vmax=None):
    if vmin is None: vmin = np.nanpercentile(data_array, 2)
    if vmax is None: vmax = np.nanpercentile(data_array, 98)
    
    data_array = np.nan_to_num(data_array, nan=vmin)
    data_array = np.clip(data_array, vmin, vmax)

    height, width = data_array.shape
    fig = plt.figure(frameon=False)
    dpi = 100
    fig.set_size_inches(width / dpi, height / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data_array, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.savefig(output_path, dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close(fig)

def repair_single_file_single_channel(
        model_path,
        target_channel,
        nc_root,
        file_id_to_process,
        output_path_repaired,
        output_path_original,
        strength=0.1,
        visualize_steps=False
):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return

    model = UNet2DModel.from_pretrained(model_path).to(DEVICE)
    model.eval()

    noise_scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS)

    num_input_channels = 1 + len(CONTEXT_VARIABLES)
    
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.5] * num_input_channels, std=[0.5] * num_input_channels)
    ])

    nc_filename = f"{file_id_to_process}_prep.nc"
    nc_filepath = os.path.join(nc_root, nc_filename)
    if not os.path.exists(nc_filepath):
        print(f"Error: Input file not found: {nc_filepath}")
        return

    try:
        with nc.Dataset(nc_filepath, 'r') as dataset:
            if target_channel not in dataset.variables:
                print(f"Error: Channel {target_channel} missing.")
                return
            
            raw_target_data = dataset.variables[target_channel][:].filled(0)
            
            vis_vmin = np.nanpercentile(raw_target_data, 2)
            vis_vmax = np.nanpercentile(raw_target_data, 98)

            save_array_as_pure_png(raw_target_data, output_path_original, vmin=vis_vmin, vmax=vis_vmax)

            context_data_list = []
            ref_shape = raw_target_data.shape
            for var_name in CONTEXT_VARIABLES:
                if var_name in dataset.variables:
                    data = dataset.variables[var_name][:].filled(0)
                else:
                    data = np.zeros(ref_shape, dtype=np.float32)
                context_data_list.append(torch.from_numpy(data).unsqueeze(0))

            target_tensor = torch.from_numpy(raw_target_data).unsqueeze(0)

        full_tensor_raw = torch.cat([target_tensor, *context_data_list], dim=0)
        
        full_tensor_processed = preprocess(full_tensor_raw.unsqueeze(0))
        
        full_tensor_processed = full_tensor_processed.to(DEVICE)

        noisy_btemp_tensor = full_tensor_processed[:, 0:1, :, :] 
        context_tensor = full_tensor_processed[:, 1:, :, :]

        start_timestep = int(noise_scheduler.config.num_train_timesteps * strength)
        start_timestep = min(start_timestep, noise_scheduler.config.num_train_timesteps - 1)
        
        timesteps = torch.tensor([start_timestep], device=DEVICE).long()
        noise = torch.randn_like(noisy_btemp_tensor)
        
        latents = noise_scheduler.add_noise(noisy_btemp_tensor, noise, timesteps)

        timesteps_to_run = noise_scheduler.timesteps[noise_scheduler.timesteps <= start_timestep]

        with torch.no_grad():
            for t in tqdm(timesteps_to_run, desc=f"Repairing {file_id_to_process}"):
                model_input = torch.cat([latents, context_tensor], dim=1)
                
                predicted_noise = model(model_input, t).sample
                
                latents = noise_scheduler.step(predicted_noise, t, latents).prev_sample

        repaired_image = latents.clamp(-1, 1) * 0.5 + 0.5
        
        repaired_image_np = repaired_image.squeeze().cpu().numpy()

        save_array_as_pure_png(repaired_image_np, output_path_repaired)

        print(f"Saved: {output_path_repaired}")

    except Exception as e:
        print(f"Failed to process {file_id_to_process}: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    NC_ROOT_DIR = 'ready-to-train_train'
    MODEL_PATH = "./diff_model/best_model" 
    BASE_OUTPUT_DIR = "./inference_output"
    FILE_ID_TO_FIX = "20180108T184332_dmi"
    TARGET_CHANNEL = 'btemp_6_9h'

    for seed in range(5):
        set_seed(seed)
        
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)
        
        out_repaired = os.path.join(output_dir, f"{FILE_ID_TO_FIX}_repaired.png")
        out_original = os.path.join(output_dir, f"{FILE_ID_TO_FIX}_original.png")

        repair_single_file_single_channel(
            model_path=MODEL_PATH,
            target_channel=TARGET_CHANNEL,
            nc_root=NC_ROOT_DIR,
            file_id_to_process=FILE_ID_TO_FIX,
            output_path_repaired=out_repaired,
            output_path_original=out_original,
            strength=0.6
        )