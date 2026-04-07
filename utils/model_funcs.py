from joblib import load
import numpy as np
import rasterio

def build_features_from_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read().astype("float32")
        valid_mask = src.read_masks(1) > 0

    blue, green, red = arr

    eps = 1e-6
    ndwi = (green - red) / (green + red + eps)
    brightness = (red + green + blue) / 3.0

    stack = np.stack([blue, green, red, ndwi, brightness], axis=0)

    return stack, valid_mask


def classify_image(tif_path, model, out_path=None):
    stack, valid_mask = build_features_from_rgb(tif_path)

    bands, h, w = stack.shape

    X = stack.reshape(bands, -1).T
    valid = valid_mask.reshape(-1)

    pred = np.zeros(len(valid), dtype=np.uint8)
    pred[valid] = model.predict(X[valid])

    mask = pred.reshape(h, w)

    if out_path:
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()

        profile.update(count=1, dtype=rasterio.uint8, nodata=0)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask, 1)

    return mask

def water_area_km2(mask, pixel_size_m=150):
    pixel_area_m2 = pixel_size_m ** 2
    return mask.sum() * pixel_area_m2 / 1e6