"""
Google Earth Engine data acquisition for Sentinel-2 imagery.

Downloads and processes Sentinel-2 SR (Surface Reflectance) data with
cloud masking and NDVI extraction.

Pipeline:
    get_s2_sr_cld_col()        → join S2_SR with S2_CLOUD_PROBABILITY
    add_cloud_bands()          → add 'clouds' band (probability threshold)
    add_shadow_bands()         → add 'dark_pixels', 'cld_proj', 'shadows' bands
    add_cld_shdw_mask()        → combine + dilate → 'cld_shdw_mask' band
    apply_cld_shdw_mask()      → apply mask to B.* bands

Example:
    >>> import ee
    >>> ee.Initialize(project='mapbiomas')
    >>> col = get_s2_sr_cld_col(aoi, '2021-01-01', '2022-12-31')
    >>> col_masked = get_s2_sr_cld_col_masked(aoi, '2021-01-01', '2022-12-31')
    >>> ndvi_series = extract_ndvi_series(col_masked, aoi)

Reference:
    - Sentinel-2: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    - Cloud masking: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-cloud-masking
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def get_s2_sr_cld_col(
    aoi: "ee.Geometry",
    start_date: str,
    end_date: str,
    cloud_filter: int = 30,
) -> "ee.ImageCollection":
    """
    Load Sentinel-2 SR collection with cloud probability layer joined.

    Parameters:
    -----------
    aoi : ee.Geometry
        Area of interest (point, polygon, etc)
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    cloud_filter : int
        Maximum cloud cover percentage (0-100)

    Returns:
    --------
    ee.ImageCollection
        Sentinel-2 SR with S2_CLOUD_PROBABILITY joined as 's2_cloudless' property
    """
    import ee

    # Main Sentinel-2 SR collection
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
    )

    # Cloud probability collection
    s2_cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join by system:index — use **{} dict unpacking for EE Python API compatibility
    joined = ee.ImageCollection(
        ee.Join.saveFirst("s2_cloudless").apply(**{
            "primary": s2_sr,
            "secondary": s2_cloudless,
            "condition": ee.Filter.equals(**{
                "leftField": "system:index",
                "rightField": "system:index",
            }),
        })
    )

    logger.info(
        f"Loaded S2 SR collection ({start_date} → {end_date}, "
        f"cloud_filter≤{cloud_filter}%)"
    )

    return joined


def add_cloud_bands(
    image: "ee.Image", cld_prb_thresh: int = 30
) -> "ee.Image":
    """
    Add cloud mask band based on S2_CLOUD_PROBABILITY.

    Parameters:
    -----------
    image : ee.Image
        S2 SR image with 's2_cloudless' property (from get_s2_sr_cld_col)
    cld_prb_thresh : int
        Cloud probability threshold (0-100). Pixels above → cloud.

    Returns:
    --------
    ee.Image
        Image with 'clouds' band added (1=cloud, 0=clear)
    """
    import ee

    cld_prb = ee.Image(image.get("s2_cloudless")).select("probability")
    is_cloud = cld_prb.gt(cld_prb_thresh).rename("clouds")
    return image.addBands(is_cloud)


def add_shadow_bands(
    image: "ee.Image",
    nir_drk_thresh: float = 0.15,
    cld_prj_dist: int = 1,
) -> "ee.Image":
    """
    Add shadow mask bands using geometric projection from cloud positions.

    Approach (from GEE cloud masking tutorial):
    1. Detect dark NIR pixels (< nir_drk_thresh × 1e4 reflectance) not over water
    2. Project 'clouds' band in sun-opposite direction using directionalDistanceTransform
    3. Shadow = dark pixels that overlap with the projected cloud mask

    Parameters:
    -----------
    image : ee.Image
        S2 SR image with 'clouds' band (from add_cloud_bands)
    nir_drk_thresh : float
        NIR darkness threshold in [0, 1] reflectance (SR scale ×1e4)
        e.g. 0.15 → pixels with B8 < 1500 are considered dark
    cld_prj_dist : int
        Shadow projection distance (× 10 pixels = × 100m)

    Returns:
    --------
    ee.Image
        Image with 'dark_pixels', 'cld_proj', and 'shadows' bands added
    """
    import ee

    SR_BAND_SCALE = 1e4

    # Dark NIR pixels that are not water (SCL=6) — potential shadow
    not_water = image.select("SCL").neq(6)
    dark_pixels = (
        image.select("B8")
        .lt(nir_drk_thresh * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Shadow azimuth: 90° − solar azimuth (projects opposite to sun)
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project cloud mask in shadow direction at 100m scale (speed/accuracy trade-off)
    cld_proj = (
        image.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, cld_prj_dist * 10)
        .reproject(**{"crs": image.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cld_proj")
    )

    # Shadow = dark pixels that intersect with the cloud projection
    shadows = dark_pixels.And(cld_proj).rename("shadows")

    return image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(
    image: "ee.Image",
    buffer: int = 50,
) -> "ee.Image":
    """
    Combine cloud and shadow masks, remove small patches, and dilate.

    Parameters:
    -----------
    image : ee.Image
        S2 SR image with 'clouds' and 'shadows' bands
    buffer : int
        Dilation buffer in meters (applied after focalMin cleanup)

    Returns:
    --------
    ee.Image
        Image with 'cld_shdw_mask' band added (1=masked, 0=valid)

    Note:
        Uses 20m scale internally for speed. The focalMax radius in pixels
        at 20m scale = buffer_meters * 2 / 20.
    """
    import ee

    # Combine: any pixel flagged as cloud OR shadow
    is_cld_shdw = image.select("clouds").add(image.select("shadows")).gt(0)

    # Remove isolated small patches (focalMin radius=2px at 20m) then dilate
    is_cld_shdw = (
        is_cld_shdw
        .focalMin(2)
        .focalMax(buffer * 2 / 20)
        .reproject(**{"crs": image.select(0).projection(), "scale": 20})
        .rename("cld_shdw_mask")
    )

    return image.addBands(is_cld_shdw)


def apply_cld_shdw_mask(image: "ee.Image") -> "ee.Image":
    """
    Apply cloud/shadow mask to reflectance bands, keeping only B.* bands.

    Parameters:
    -----------
    image : ee.Image
        S2 SR image with 'cld_shdw_mask' band

    Returns:
    --------
    ee.Image
        Masked image with only B.* bands (masked pixels = NaN)
    """
    import ee

    mask = image.select("cld_shdw_mask").eq(0)
    return image.select("B.*").updateMask(mask)


def get_s2_sr_cld_col_masked(
    aoi: "ee.Geometry",
    start_date: str,
    end_date: str,
    cloud_filter: int = 30,
    cld_prb_thresh: int = 30,
    nir_drk_thresh: float = 0.15,
    cld_prj_dist: int = 1,
    buffer: int = 50,
) -> "ee.ImageCollection":
    """
    Complete pipeline: load Sentinel-2, add masks, apply masking.

    Chains all steps: get_s2_sr_cld_col → add_cloud_bands →
    add_shadow_bands → add_cld_shdw_mask → apply_cld_shdw_mask.

    Parameters:
    -----------
    aoi : ee.Geometry
        Area of interest
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    cloud_filter : int
        Max cloud cover % for initial scene filter
    cld_prb_thresh : int
        Cloud probability threshold
    nir_drk_thresh : float
        NIR darkness threshold for shadow detection
    cld_prj_dist : int
        Shadow projection distance (× 100m)
    buffer : int
        Dilation buffer (meters)

    Returns:
    --------
    ee.ImageCollection
        Cloud/shadow-masked Sentinel-2 SR (only B.* bands)
    """
    col = get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter)
    col = col.map(lambda img: add_cloud_bands(img, cld_prb_thresh))
    col = col.map(lambda img: add_shadow_bands(img, nir_drk_thresh, cld_prj_dist))
    col = col.map(lambda img: add_cld_shdw_mask(img, buffer))
    col = col.map(apply_cld_shdw_mask)
    return col


def extract_ndvi_series(
    collection: "ee.ImageCollection",
    aoi: "ee.Geometry",
    scale: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract NDVI time series from a masked Sentinel-2 collection.

    Maps over the collection server-side to compute mean NDVI per image,
    then downloads the result as a FeatureCollection in a single getInfo() call.

    Parameters:
    -----------
    collection : ee.ImageCollection
        Cloud-masked S2 collection (output of apply_cld_shdw_mask)
    aoi : ee.Geometry
        Area of interest
    scale : int
        Scale in meters (10 = S2 native resolution)

    Returns:
    --------
    List[Dict]
        Sorted list of {date, ndvi} observations:
        [{"date": "2021-01-01", "ndvi": 0.82}, ...]
    """
    import ee

    def to_ndvi_feature(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
        )
        # Count valid (non-masked) pixels vs total pixels in AOI
        valid_px = img.select("B4").reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
        )
        total_px = img.select("B4").unmask(0).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
        )
        return ee.Feature(None, {
            "NDVI": val.get("NDVI"),
            "date": img.date().format("YYYY-MM-dd"),
            "valid_px": valid_px.get("B4"),
            "total_px": total_px.get("B4"),
        })

    fc = ee.FeatureCollection(collection.map(to_ndvi_feature))
    features = fc.getInfo()["features"]

    series = []
    for feat in features:
        props = feat["properties"]
        ndvi = props.get("NDVI")
        date = props.get("date")
        if date and ndvi is not None:
            valid_px = props.get("valid_px") or 0
            total_px = props.get("total_px") or 1
            valid_pct = round(valid_px / total_px, 4) if total_px > 0 else 0.0
            series.append({"date": date, "ndvi": round(float(ndvi), 4), "valid_pct": valid_pct})

    series.sort(key=lambda x: x["date"])
    logger.info(f"Extracted {len(series)} NDVI observations")
    return series


def download_s2_scene(
    image: "ee.Image",
    aoi: "ee.Geometry",
    folder: str = "ee-mapbiomas",
    filename: str = "s2_scene",
    scale: int = 10,
) -> str:
    """
    Download a single S2 scene to Google Drive.

    Parameters:
    -----------
    image : ee.Image
        S2 image to download
    aoi : ee.Geometry
        Area of interest (clipping region)
    folder : str
        Google Drive folder name
    filename : str
        Output filename (without extension)
    scale : int
        Scale in meters

    Returns:
    --------
    str
        Task ID (for monitoring)
    """
    import ee

    image_clipped = image.clip(aoi)

    task = ee.batch.Export.image.toDrive(
        image=image_clipped,
        description=filename,
        folder=folder,
        fileNamePrefix=filename,
        scale=scale,
        maxPixels=1e13,
    )

    task.start()
    logger.info(f"Started download: {filename} (task_id={task.id})")
    return task.id


def get_ndvi_series_for_aoi(
    aoi_geometry,
    start_date: str,
    end_date: str,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    All-in-one: get masked NDVI time series for an AOI.

    Parameters:
    -----------
    aoi_geometry : ee.Geometry
        Area of interest
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    **kwargs : dict
        Optional overrides: cloud_filter, cld_prb_thresh, nir_drk_thresh,
        cld_prj_dist, buffer, scale

    Returns:
    --------
    List[Dict]
        NDVI time series [{date, ndvi}, ...]
    """
    col = get_s2_sr_cld_col_masked(
        aoi_geometry,
        start_date,
        end_date,
        cloud_filter=kwargs.get("cloud_filter", 30),
        cld_prb_thresh=kwargs.get("cld_prb_thresh", 30),
        nir_drk_thresh=kwargs.get("nir_drk_thresh", 0.15),
        cld_prj_dist=kwargs.get("cld_prj_dist", 1),
        buffer=kwargs.get("buffer", 50),
    )
    return extract_ndvi_series(col, aoi_geometry, scale=kwargs.get("scale", 10))
