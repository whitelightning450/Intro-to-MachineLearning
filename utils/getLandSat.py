import os
from datetime import date, datetime
import ee
import geemap
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

ee.Authenticate()
ee.Initialize()
print("Earth Engine initialized successfully.")


def get_bbox(bbox):
    return ee.Geometry.Rectangle(bbox)


def apply_scale_factors(image):
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


#making a looser mask
def mask_landsat_c2_l2(image):
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0)  # cloud only
    return image.updateMask(mask)


def prep_l57(image):
    image = apply_scale_factors(image)
    image = mask_landsat_c2_l2(image)
    return image.select(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        ["blue", "green", "red", "nir", "swir1", "swir2"]
    ).copyProperties(image, image.propertyNames())


def prep_l89(image):
    image = apply_scale_factors(image)
    image = mask_landsat_c2_l2(image)
    return image.select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        ["blue", "green", "red", "nir", "swir1", "swir2"]
    ).copyProperties(image, image.propertyNames())
    
    
def get_landsat5_collection(start_date, end_date, roi):
    return (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .map(prep_l57)
    )

def get_landsat7_collection(start_date, end_date, roi):
    return (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .map(prep_l57)
    )


def get_landsat8_collection(start_date, end_date, roi):
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .map(prep_l89)
    )


def _as_py_date(d):
    return datetime.strptime(d, "%Y-%m-%d").date()


def get_landsat_collection(start_date, end_date, roi):
    start_py = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_py = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Landsat 4/5 era
    if end_py < date(1999, 5, 28):
        return get_landsat5_collection(start_date, end_date, roi)

    # Landsat 7 era
    if start_py >= date(1999, 5, 28) and end_py < date(2013, 3, 18):
        return get_landsat7_collection(start_date, end_date, roi)

    # Landsat 8+ era
    if start_py >= date(2013, 3, 18):
        return get_landsat8_collection(start_date, end_date, roi)

    # Ranges that cross sensor boundaries
    collections = []

    if start_py < date(1999, 5, 28):
        collections.append(get_landsat5_collection(start_date, "1999-05-28", roi))

    if start_py < date(2013, 3, 18) and end_py >= date(1999, 5, 28):
        collections.append(get_landsat7_collection(
            max(start_date, "1999-05-28"), "2013-03-18", roi
        ))

    if end_py >= date(2013, 3, 18):
        collections.append(get_landsat8_collection(
            max(start_date, "2013-03-18"), end_date, roi
        ))

    out = collections[0]
    for c in collections[1:]:
        out = out.merge(c)
    return out


def make_monthly_composite(start_date, end_date, roi):
    collection = get_landsat_collection(start_date, end_date, roi)

    if collection.size().getInfo() == 0:
        print(f"No images for {start_date} to {end_date}")
        return None

    return collection.median().clip(roi)

def make_rolling_composite(center_date, roi, months_before=1, months_after=1, reducer="median"):
    """
    Build a rolling composite centered on center_date.

    months_before=1, months_after=1 gives a 3-month window:
    previous month, center month, next month.
    """
    center = ee.Date(center_date)
    start = center.advance(-months_before, "month")
    end = center.advance(months_after, "month")

    collection = get_landsat_collection(
        start.format("YYYY-MM-dd").getInfo(),
        end.format("YYYY-MM-dd").getInfo(),
        roi
    )

    if collection.size().getInfo() == 0:
        print(f"No images for {start.getInfo()} to {end.getInfo()}")
        return None

    if reducer == "mean":
        img = collection.mean()
    else:
        img = collection.median()

    return img.clip(roi)


#function adding rolling window to composite
def export_monthly_composite(
    center_date,
    roi,
    output_dir="data/LandSat/GSL",
    filename="GSL_monthly_composite.tif",
    scale=150,
    months_before=1,
    months_after=1,
    reducer="median"
):
    os.makedirs(output_dir, exist_ok=True)

    img = make_rolling_composite(
        center_date=center_date,
        roi=roi,
        months_before=months_before,
        months_after=months_after,
        reducer=reducer
    )

    if img is None:
        return
    
    # select RGB bands
    img = img.select(["red", "green", "blue"])

    # APPLY ENHANCEMENT
    img = enhance_image(img, roi)
    out_path = os.path.join(output_dir, filename)

    geemap.ee_export_image(
        img.select(["red", "green", "blue"]),
        filename=out_path,
        scale=scale,
        region=roi,
        file_per_band=False
    )

    print(f"Saved to {out_path}")
    

#function to enhance contrast of image using 2nd and 98th percentiles
def enhance_image(img, roi):
    bands = ["red", "green", "blue"]

    stats = img.select(bands).reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=roi,
        scale=150,
        maxPixels=1e13
    )

    def stretch(band):
        minv = ee.Number(stats.get(f"{band}_p2"))
        maxv = ee.Number(stats.get(f"{band}_p98"))
        return img.select(band).subtract(minv).divide(maxv.subtract(minv))

    stretched = ee.Image.cat([stretch(b) for b in bands])
    return stretched.clamp(0, 1)
    
    
#parrallel export function
def month_starts(start_year=1985, end_year=2025):
    """
    Yield (start_date, end_date, filename) for each month.
    """
    y = start_year
    m = 1

    while y <= end_year:
        start = date(y, m, 1)

        # compute next month
        if m == 12:
            next_start = date(y + 1, 1, 1)
        else:
            next_start = date(y, m + 1, 1)

        yield (
            start.strftime("%Y-%m-%d"),
            next_start.strftime("%Y-%m-%d"),
            f"GSL_{y}_{m:02d}.tif"
        )

        # increment month
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def export_monthly_range_parallel(
    roi,
    output_dir="data/LandSat/GSL",
    start_year=1985,
    end_year=2025,
    max_workers=3
):
    """
    Export monthly composites between start_year and end_year.
    """
    os.makedirs(output_dir, exist_ok=True)

    jobs = list(month_starts(start_year=start_year, end_year=end_year))

    print(f"Prepared {len(jobs)} monthly jobs ({start_year}–{end_year})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(export_monthly_composite, s, e, roi, output_dir, fname)
            for s, e, fname in jobs
        ]

        for fut in as_completed(futures):
            try:
                result = fut.result()
                print(f"Finished: {result}")
            except Exception as e:
                print(f"Failed: {e}")

    print("All monthly exports completed.")


def month_centers(start_year=1985, end_year=2025):
    """
    Yield the first day of each month as YYYY-MM-01.
    """
    y = start_year
    m = 1

    while y <= end_year:
        yield f"{y}-{m:02d}-01"

        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
            
def make_rolling_composite(center_date, roi, months_before=1, months_after=1, reducer="median"):
    center = ee.Date(center_date)
    start = center.advance(-months_before, "month")
    end = center.advance(months_after + 1, "month")

    start_str = start.format("YYYY-MM-dd").getInfo()
    end_str = end.format("YYYY-MM-dd").getInfo()

    collection = get_landsat_collection(start_str, end_str, roi)

    if collection.size().getInfo() == 0:
        print(f"No images for {start_str} to {end_str}")
        return None

    if reducer == "mean":
        img = collection.mean()
    else:
        img = collection.median()

    return img.clip(roi)

def export_rolling_composite(
    center_date,
    roi,
    output_dir="data/LandSat/GSL",
    filename=None,
    scale=150,
    months_before=1,
    months_after=1,
    reducer="median"
):
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        y, m, _ = center_date.split("-")
        filename = f"GSL_{y}_{m}.tif"

    img = make_rolling_composite(
        center_date=center_date,
        roi=roi,
        months_before=months_before,
        months_after=months_after,
        reducer=reducer
    )

    if img is None:
        return None
    
    # select RGB bands
    img = img.select(["red", "green", "blue"])

    # APPLY ENHANCEMENT
    img = enhance_image(img, roi)

    out_path = os.path.join(output_dir, filename)

    geemap.ee_export_image(
        img.select(["red", "green", "blue"]),
        filename=out_path,
        scale=scale,
        region=roi,
        file_per_band=False
    )

    print(f"Saved to {out_path}")
    return out_path


def export_monthly_range_parallel(
    roi,
    output_dir="data/LandSat/GSL",
    start_year=1985,
    end_year=2025,
    max_workers=3,
    months_before=1,
    months_after=1,
    reducer="median"
):
    os.makedirs(output_dir, exist_ok=True)

    jobs = list(month_centers(start_year=start_year, end_year=end_year))
    print(f"Prepared {len(jobs)} monthly jobs")

    def run_job(center_date):
        y, m, _ = center_date.split("-")
        filename = f"GSL_{y}_{m}.tif"
        return export_rolling_composite(
            center_date=center_date,
            roi=roi,
            output_dir=output_dir,
            filename=filename,
            scale=150,
            months_before=months_before,
            months_after=months_after,
            reducer=reducer
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, center_date) for center_date in jobs]

        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result:
                    print(f"Finished: {result}")
            except Exception as e:
                print(f"Failed: {e}")

    print("All monthly exports completed.")