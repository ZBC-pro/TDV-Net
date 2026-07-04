import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import os
import glob
import json

INPUT_DIRECTORY = 'ready-to-train_train'
JSON_SPLIT_FILE = 'diff_train/dataset_split.json'
BASE_VIS_OUTPUT_DIR = 'ready-to-train_train_visualization'
MASK_OUTPUT_DIR = 'ready-to-train_train_masks'
TARGET_VARIABLES = ['btemp_6_9h', 'btemp_6_9v']
DETECTION_VARIABLE = 'btemp_6_9h'
CANNY_SIGMA = 1.5
LINE_THICKNESS = 2.0

os.makedirs(BASE_VIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

try:
    with open(JSON_SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    noisy_prefixes = set(split_data.get('noisy_files', []))
    clean_prefixes = set(split_data.get('clean_files', []))
except Exception as e:
    print(f"JSON load error: {e}")
    noisy_prefixes = set()
    clean_prefixes = set()

nc_files = glob.glob(os.path.join(INPUT_DIRECTORY, '*.nc'))
if not nc_files:
    print("No NC files found.")
    exit()

for i, nc_path in enumerate(nc_files):
    base_name = os.path.basename(nc_path)
    print(f"Processing ({i+1}/{len(nc_files)}): {base_name}")
    try:
        with nc.Dataset(nc_path, 'r') as src:
            if DETECTION_VARIABLE not in src.variables:
                print(f"  Skipping: {DETECTION_VARIABLE} not found.")
                continue
            data_raw = src.variables[DETECTION_VARIABLE][:]
            data_filled = np.nan_to_num(data_raw)
    except Exception as e:
        print(f"  Error reading data: {e}")
        continue

    try:
        edges = canny(data_filled, sigma=CANNY_SIGMA)
        h, theta, d = hough_line(edges)
        accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=1)
        line_detected = len(angles) > 0
        if line_detected:
            angle = angles[0]
            dist = dists[0]
        else:
            angle = 0.0
            dist = 0.0
            print("  No prominent line detected.")
    except Exception as e:
        print(f"  Hough transform error: {e}")
        edges = np.zeros_like(data_filled)
        line_detected = False
        angle = 0.0
        dist = 0.0

    file_prefix = '_'.join(base_name.split('_')[:2])
    if file_prefix in noisy_prefixes:
        category = 'noisy'
    elif file_prefix in clean_prefixes:
        category = 'clean'
    else:
        category = 'other'

    if line_detected:
        seam_mask = np.zeros_like(data_raw, dtype=np.int8)
        y_indices, x_indices = np.indices(data_raw.shape)
        line_check = np.abs(x_indices * np.cos(angle) + y_indices * np.sin(angle) - dist)
        seam_mask[line_check < LINE_THICKNESS] = 1
        try:
            mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name.replace('.nc', '_hough_mask.nc'))
            with nc.Dataset(nc_path, 'r') as src_copy:
                with nc.Dataset(mask_save_path, 'w', format='NETCDF4') as dst:
                    dst.setncatts(src_copy.__dict__)
                    for name, dimension in src_copy.dimensions.items():
                        dst.createDimension(name, len(dimension))
                    for name, variable in src_copy.variables.items():
                        out_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                        out_var.setncatts(src_copy[name].__dict__)
                        out_var[:] = src_copy[name][:]
                    mask_var = dst.createVariable('hough_seam_mask', 'i1', src_copy[DETECTION_VARIABLE].dimensions)
                    mask_var.long_name = 'Detected seam line using Hough Transform'
                    mask_var.description = f'Line parameters (angle, dist): ({angle}, {dist})'
                    mask_var[:] = seam_mask
            print(f"  Mask saved to {mask_save_path}")
        except Exception as e:
            print(f"  Error saving mask NC: {e}")

    category_dir = os.path.join(BASE_VIS_OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    axes[0].imshow(data_raw, cmap='viridis')
    axes[0].set_title(f'Original {DETECTION_VARIABLE}')
    axes[0].set_axis_off()
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Canny Edge Map')
    axes[1].set_axis_off()
    axes[2].imshow(data_raw, cmap='viridis')
    if line_detected:
        if abs(np.sin(angle)) > 1e-6:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - edges.shape[1] * np.cos(angle)) / np.sin(angle)
            axes[2].plot((0, edges.shape[1]), (y0, y1), '-r', linewidth=2)
        else:
            x0 = dist / np.cos(angle) if abs(np.cos(angle)) > 1e-6 else 0
            axes[2].axvline(x=x0, color='r', linewidth=2)
    axes[2].set_xlim(0, edges.shape[1])
    axes[2].set_ylim(edges.shape[0], 0)
    axes[2].set_title('Hough Line Detection Result')
    axes[2].set_axis_off()
    plt.tight_layout()
    vis_save_name = base_name.replace('.nc', '_detection.png')
    vis_save_path = os.path.join(category_dir, vis_save_name)
    plt.savefig(vis_save_path, dpi=150)
    plt.close(fig)

    for var_name in TARGET_VARIABLES:
        if var_name == DETECTION_VARIABLE:
            continue
        try:
            with nc.Dataset(nc_path, 'r') as src:
                if var_name in src.variables:
                    data_var = src.variables[var_name][:]
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    img = ax2.imshow(data_var, cmap='viridis')
                    ax2.set_title(f'{var_name}')
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    plt.colorbar(img, ax=ax2)
                    plt.tight_layout()
                    var_save_name = base_name.replace('.nc', f'_{var_name}.png')
                    var_save_path = os.path.join(category_dir, var_save_name)
                    plt.savefig(var_save_path, dpi=150)
                    plt.close(fig2)
                else:
                    print(f"  Variable {var_name} not found, skipping visualization.")
        except Exception as e:
            print(f"  Error visualizing {var_name}: {e}")

print("Batch processing complete.")