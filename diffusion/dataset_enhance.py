import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import netCDF4 as nc
import numpy as np
import os
import json
from skimage.draw import polygon
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

BTEMP_VARIABLES = [
    'btemp_6_9h', 
    'btemp_6_9v', 
    'btemp_7_3h', 'btemp_7_3v', 
    'btemp_10_7h','btemp_10_7v', 
    'btemp_18_7h', 'btemp_18_7v', 
    'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v', 
]

CONTEXT_VARIABLES = [
    'skt', 't2m', 
    'tcwv', 'u10m_rotated', 'v10m_rotated'
]

def create_seamline_from_single_image(image, max_contrast_delta=0.06, max_brightness_delta=0.06, max_bend_factor=0.1):
    height, width = image.shape
    if height == 0 or width == 0: return image
    if np.random.rand() > 0.5:
        start_x, start_y = np.random.randint(0, width * 0.6), 0
        end_x, end_y = width - 1, np.random.randint(height * 0.4, height)
    else:
        start_x, start_y = 0, np.random.randint(0, height * 0.6)
        end_x, end_y = np.random.randint(width * 0.4, width), height - 1
    mid_x = (start_x + end_x) / 2 + np.random.randint(-width * max_bend_factor, width * max_bend_factor)
    mid_y = (start_y + end_y) / 2 + np.random.randint(-height * max_bend_factor, height * max_bend_factor)
    px, py = np.array([start_x, mid_x, end_x]), np.array([start_y, mid_y, end_y])
    polygon_points = np.vstack(
        [np.vstack([px, py]).T, [width, end_y], [width, height], [-1, height], [-1, start_y], [start_x, -1]])
    mask = np.zeros_like(image, dtype=np.float32)
    rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0], shape=image.shape)
    mask[rr, cc] = 1.0
    image_altered = image.copy()
    contrast_factor = 1.0 + np.random.uniform(-max_contrast_delta, max_contrast_delta)
    dynamic_range = np.max(image) - np.min(image) if np.max(image) > np.min(image) else 1
    brightness_factor = np.random.uniform(-max_brightness_delta, max_brightness_delta) * dynamic_range
    image_altered[mask == 1] = image_altered[mask == 1] * contrast_factor + brightness_factor
    min_val, max_val = np.min(image), np.max(image)
    image_altered = np.clip(image_altered, min_val, max_val)
    return image_altered.astype(image.dtype)

class NCDataset(Dataset):
    def __init__(self, nc_root, json_path, split, transform=None, context_transform=None):
        self.nc_root = nc_root
        self.transform = transform
        self.context_transform = context_transform
        self.samples = []

        try:
            with open(json_path, 'r') as f:
                split_data = json.load(f)

            key = "clean_files" if split == 'train' else "validation_files"
            file_ids = split_data.get(key, [])

            for file_id in file_ids:
                for target_var in BTEMP_VARIABLES:
                    self.samples.append((file_id, target_var))

        except Exception as e:
            print(f"Warning: Could not load or parse JSON file at {json_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id, target_btemp_var = self.samples[idx]
        nc_filename = f"{file_id}_prep.nc"
        nc_filepath = os.path.join(self.nc_root, nc_filename)

        try:
            with nc.Dataset(nc_filepath, 'r') as dataset:
                ref_shape = (dataset.dimensions['2km_grid_lines'].size, dataset.dimensions['2km_grid_samples'].size)

                clean_target_data = dataset.variables[target_btemp_var][:].filled(np.nan)
                damaged_target_data = create_seamline_from_single_image(clean_target_data)

                context_channels_data = []
                for var_name in CONTEXT_VARIABLES:
                    data = dataset.variables[var_name][:].filled(np.nan) if var_name in dataset.variables else np.zeros(
                        ref_shape, dtype=np.float32)
                    context_channels_data.append(data)

                clean_target_tensor = torch.from_numpy(np.nan_to_num(clean_target_data)).unsqueeze(0)
                damaged_target_tensor = torch.from_numpy(np.nan_to_num(damaged_target_data)).unsqueeze(0)
                context_tensor = torch.from_numpy(np.nan_to_num(np.stack(context_channels_data, axis=0)))

            if self.transform:
                full_damaged_tensor = torch.cat([damaged_target_tensor, context_tensor], dim=0)
                full_clean_tensor = torch.cat([clean_target_tensor, context_tensor], dim=0)

                full_damaged_transformed = self.transform(full_damaged_tensor)
                full_clean_transformed = self.transform(full_clean_tensor)

                damaged_input = full_damaged_transformed
                clean_target = full_clean_transformed[0:1, :, :]

            return {"damaged_input": damaged_input, "clean_target": clean_target, "id": file_id}

        except Exception as e:
            return None

def get_dataloader(nc_root, json_path, split, image_size, batch_size, use_augmentation=False, num_workers=0):
    num_input_channels = 1 + len(CONTEXT_VARIABLES)

    if use_augmentation and split == 'train':
        print("Using data augmentation for training set.")
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize(mean=[0.5] * num_input_channels, std=[0.5] * num_input_channels)
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.5] * num_input_channels, std=[0.5] * num_input_channels)
        ])

    dataset = NCDataset(
        nc_root=nc_root,
        json_path=json_path,
        split=split,
        transform=preprocess
    )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"DataLoader for '{split}' split created: {len(dataset)} samples.")
    return dataloader