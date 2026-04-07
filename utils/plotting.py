
import rasterio
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
import rasterio
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm



#create a function to plot landsat image
def plot_landsat_image(path, date, size=(8,8)):
    
    filepath = f"{path}GSL_{date}.tif"
    with rasterio.open(filepath) as src:
        img = src.read()  # (bands, rows, cols)

    # Convert to (rows, cols, bands)
    img = img.transpose(1, 2, 0)

    # Normalize for display (important!)
    img = img / img.max()

    plt.figure(figsize=size)
    plt.imshow(img)
    plt.title(f"Landsat Image: {date}")
    plt.axis("off")
    plt.show()
    
    # Make gifs from tifs
def make_gif_from_tifs(
    input_dir,
    output_gif="data/LandSat/GSL/gsl_timelapse.gif",
    pattern="GSL_*.tif",
    scale_max=0.3,
    frame_duration=0.6,
    max_width=800,
    font_size=56
):
    tif_files = glob.glob(os.path.join(input_dir, pattern))
    tif_files = sorted(tif_files, key=natural_sort_key)

    if not tif_files:
        raise FileNotFoundError(f"No TIFFs found in {input_dir} matching {pattern}")

    frames = []

    for tif in tqdm(tif_files, desc="Building GIF", unit="frame"):
        img = tif_to_rgb_array(tif, scale_max=scale_max)

        h, w = img.shape[:2]
        if w > max_width:
            new_h = int(h * max_width / w)
            img = np.array(Image.fromarray(img).resize((max_width, new_h)))

        date_text = extract_date_from_filename(tif)
        img = add_text_overlay(img, date_text, font_size=font_size)

        frames.append(img)

    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    
    print(f"Saving GIF to {output_gif} with frame duration {frame_duration}s...")
   # imageio.mimsave(output_gif, frames, duration=frame_duration)
    save_gif_pillow(frames, output_gif, frame_duration_s=frame_duration, loop=0)

    print(f"Saved GIF to {output_gif}")

def save_gif_pillow(frames, output_gif, frame_duration_s=4.0, loop=0):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    frame_duration_ms = int(frame_duration_s * 1000)

    pil_frames[0].save(
        output_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=loop,
        disposal=2
    )
    
    
def natural_sort_key(path):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def extract_date_from_filename(path):
    name = os.path.basename(path)
    match = re.search(r"(\d{4})_(\d{2})", name)
    if match:
        year, month = match.groups()
        return f"{year}-{month}"
    return ""


def tif_to_rgb_array(tif_path, scale_max=0.3):
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3]).astype("float32")

    arr = np.moveaxis(arr, 0, -1)
    arr = np.clip(arr, 0, scale_max)
    arr = arr / scale_max
    arr = (arr * 255).astype(np.uint8)
    return arr


def add_text_overlay(img_array, text, font_size=48):
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    x, y = 20, 20
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 10
    bg = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)

    draw.rectangle(bg, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return np.array(img)


import matplotlib.pyplot as plt
import rasterio
import numpy as np


def plot_image_overlay_mask(tif_path, mask, date=""):
    # Load RGB image
    with rasterio.open(tif_path) as src:
        rgb = src.read([1, 2, 3]).astype("float32")

    rgb = np.moveaxis(rgb, 0, -1)

    # Normalize
    rgb = np.clip(rgb, 0, 0.3)
    rgb = rgb / 0.3

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(rgb)
    axes[0].set_title(f"Landsat ({date})")
    axes[0].axis("off")

    # 2Overlay
    axes[1].imshow(rgb)
    axes[1].imshow(mask, cmap="Blues", alpha=0.7)
    axes[1].set_title("Overlay")
    axes[1].axis("off")

    # 3Mask only
    axes[2].imshow(mask, cmap="Blues")
    axes[2].set_title("Water Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()