import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from IPython.display import clear_output



# Utility function to load a Landsat image and compute indices
def load_landsat_rgb(tif_path):
    """
    Works with 3-band RGB Landsat images.
    Returns feature stack + mask.
    """
    # Read the RGB bands and mask from the GeoTIFF file
    with rasterio.open(tif_path) as src:
        arr = src.read().astype("float32")
        valid_mask = src.read_masks(1) > 0
    # Check if the image has 3 bands (RGB)
    if arr.shape[0] != 3:
        raise ValueError(f"{tif_path} must have exactly 3 bands (RGB)")

    # Unpack the RGB bands
    blue, green, red = arr

    # Compute brightness and NDWI (Normalized Difference Water Index)
    eps = 1e-6

    # Approximate indices using RGB only
    ndwi = (green - red) / (green + red + eps)
    brightness = (red + green + blue) / 3.0

    # Stack the features into a single array
    stack = np.stack(
        [blue, green, red, ndwi, brightness],
        axis=0
    )

    # Define feature names for clarity
    feature_names = ["blue", "green", "red", "ndwi", "brightness"]

    return stack, valid_mask, feature_names

# Utility functions for visualizing the data and labeling samples
def stretch_rgb(rgb, vmin=0.0, vmax=0.3):
    # Stretch RGB values to [0, 1] for visualization
    rgb = np.clip(rgb, vmin, vmax)
    # Avoid division by zero if vmax == vmin
    rgb = (rgb - vmin) / (vmax - vmin)
    # Clip to [0, 1] after stretching
    return np.clip(rgb, 0, 1)


# This function displays a zoomed-out overview of the full image with a highlighted crop box, as well as a zoomed-in view of the crop area with an inner patch box. The full image and crop are visualized using RGB bands, and the boxes are drawn to indicate the areas of interest.
def show_overview_and_crop(stack, r0, c0, crop_size=100, patch_size=10, rgb_band_indices=(2, 1, 0)):
    """
    Show:
      1) zoomed-out full image with crop box
      2) zoomed-in crop with inner patch box
    """
    
    bands, h, w = stack.shape

    # Full image RGB
    full_rgb = np.moveaxis(stack[list(rgb_band_indices)], 0, -1)
    full_rgb = stretch_rgb(full_rgb, vmin=0.0, vmax=0.3)

    # Crop RGB
    crop = stack[:, r0:r0 + crop_size, c0:c0 + crop_size]
    crop_rgb = np.moveaxis(crop[list(rgb_band_indices)], 0, -1)
    crop_rgb = stretch_rgb(crop_rgb, vmin=0.0, vmax=0.3)

    # Box for inner patch inside crop
    patch_start = (crop_size - patch_size) // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Overview
    axes[0].imshow(full_rgb)
    axes[0].set_title("Overview")
    axes[0].axis("off")

    # Box for crop in overview
    overview_box = Rectangle(
        (c0, r0),
        crop_size,
        crop_size,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none"
    )
    axes[0].add_patch(overview_box)

    # Zoomed crop
    axes[1].imshow(crop_rgb)
    axes[1].set_title("Zoomed crop")
    axes[1].axis("off")

    inner_box = Rectangle(
        (patch_start, patch_start),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    axes[1].add_patch(inner_box)

    plt.tight_layout()
    plt.show()
    
    

# Load the RGB bands and valid mask from a Landsat image
def sample_labeled_patch_from_image(
    tif_path,
    crop_size=100,
    patch_size=10,
    rgb_band_indices=(2, 1, 0),
):
    # Load the image and valid mask
    stack, valid_mask, feature_names = load_landsat_rgb(tif_path)
    bands, h, w = stack.shape

    # Randomly select the top-left corner of the crop
    r0 = random.randint(0, h - crop_size)
    c0 = random.randint(0, w - crop_size)

    # 🔥 Clear previous output BEFORE showing new image
    clear_output(wait=True)

    show_overview_and_crop(
        stack,
        r0,
        c0,
        crop_size=crop_size,
        patch_size=patch_size,
        rgb_band_indices=rgb_band_indices
    )
    # Prompt the user to label the patch
    label = input("Label the red box: 1 = water, 0 = not water, s = skip: ").strip().lower()

    # Clear image AFTER labeling too (optional)
    clear_output(wait=True)

    if label == "s":
        return pd.DataFrame()

    if label not in {"0", "1"}:
        print("Invalid label. Skipping.")
        return pd.DataFrame()

    # Convert label to integer
    label = int(label)

    # Extract the patch from the center of the crop
    start = (crop_size - patch_size) // 2
    pr0 = r0 + start
    pc0 = c0 + start
    pr1 = pr0 + patch_size
    pc1 = pc0 + patch_size

    # Extract the patch and corresponding valid mask
    patch = stack[:, pr0:pr1, pc0:pc1]
    patch_valid = valid_mask[pr0:pr1, pc0:pc1]

    # Reshape the patch to a 2D array of pixels and filter by valid mask
    X = patch.reshape(bands, -1).T
    valid = patch_valid.reshape(-1)
    X = X[valid]

    # Create a DataFrame with the pixel values, label, and source file name
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = label
    df["source_file"] = os.path.basename(tif_path)

    print(f"Saved {len(df)} pixels ✔")

    return df



# This function allows the user to label a patch of a Landsat image by displaying it and asking for input. The user can label the patch as water (1) or not water (0). The function returns a DataFrame with the pixel values and their corresponding labels.
def labeling_session(
    image_folder,
    n_samples=20,
    crop_size=100,
    patch_size=10,
    out_csv="data/LandSat/GSL/training_samples.csv",
):
    # Get list of TIFF files in the specified image folder
    tif_files = sorted([str(p) for p in Path(image_folder).glob("*.tif")])

    # Check if any TIFF files were found, if not raise an error
    if not tif_files:
        raise FileNotFoundError(f"No TIFFs found in {image_folder}")

    all_rows = []

    # Loop through the number of samples to be labeled
    for i in range(n_samples):
        tif_path = random.choice(tif_files)
        print(f"\nSample {i+1}/{n_samples}: {os.path.basename(tif_path)}")

        # Call the function to sample a labeled patch from the image
        df = sample_labeled_patch_from_image(
            tif_path=tif_path,
            crop_size=crop_size,
            patch_size=patch_size
        )

        # If the DataFrame is not empty, append it to the list of all rows and print the number of pixels saved. Otherwise, print that the sample was skipped.
        if not df.empty:
            all_rows.append(df)
            print(f"Saved {len(df)} pixels.")
        else:
            print("Skipped.")

    # If any rows were collected, concatenate them into a single DataFrame, save it to a CSV file, and return the DataFrame. Otherwise, print that no labels were collected and return an empty DataFrame.
    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved labels to {out_csv}")
        return out_df

    # If no labels were collected, print a message and return an empty DataFrame.
    print("No labels collected.")
    return pd.DataFrame()