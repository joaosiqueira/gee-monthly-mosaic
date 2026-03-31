"""
Cloud masking utilities for Sentinel-2 imagery.

Advanced functions for cloud and shadow detection and removal.
These complement acquisition.py with alternative/additional masking strategies.

Methods:
1. Probability-based (S2_CLOUD_PROBABILITY)
   - Uses ESA's pre-computed cloud probability
   - Fast, reliable for cloud detection

2. Spectral-based (dark pixel detection)
   - Detects potential shadows by low NIR reflectance
   - Simple heuristic, complements probability method

3. Geometric-based (sun geometry projection)
   - Projects cloud positions in shadow direction using sun azimuth
   - More accurate but slower; requires 'clouds' band from acquisition pipeline

4. Combined (multiple methods)
   - OR combination; any method flagging a pixel → masked
"""

import logging

logger = logging.getLogger(__name__)


def cloud_probability_mask(
    image: "ee.Image",
    threshold: int = 30,
) -> "ee.Image":
    """
    Create cloud mask from S2_CLOUD_PROBABILITY layer.

    Parameters:
    -----------
    image : ee.Image
        S2 image with 's2_cloudless' property (from get_s2_sr_cld_col)
    threshold : int
        Cloud probability threshold (0-100). Pixels above → masked.
        25: aggressive (more masked), 30: balanced, 40: conservative

    Returns:
    --------
    ee.Image
        Single-band image 'cld_mask' (1=cloud, 0=clear)
    """
    import ee

    cld_prb = ee.Image(image.get("s2_cloudless")).select("probability")
    return cld_prb.gt(threshold).rename("cld_mask")


def shadow_mask_spectral(
    image: "ee.Image",
    nir_threshold: float = 0.15,
) -> "ee.Image":
    """
    Create shadow mask using low NIR reflectance as proxy.

    Shadows appear as dark pixels in NIR. This is a simple heuristic —
    for more accurate detection use acquisition.add_shadow_bands() which
    combines NIR darkness with geometric cloud projection.

    Parameters:
    -----------
    image : ee.Image
        S2 SR image (bands in ×10000 reflectance units)
    nir_threshold : float
        Normalized NIR threshold [0, 1]. Pixels with B8/(10000) < threshold
        are considered potential shadow.

    Returns:
    --------
    ee.Image
        Single-band image 'shd_mask' (1=shadow, 0=not shadow)

    Note:
        This will flag water, dark soil, and built-up areas as shadow too.
        Use in combination with cloud_probability_mask for better results.
    """
    import ee

    # B8 is in ×10000 reflectance; threshold is in [0,1] → multiply by 1e4
    dark_nir = image.select("B8").lt(nir_threshold * 1e4)
    return dark_nir.rename("shd_mask")


def shadow_mask_geometric(
    image: "ee.Image",
    cloud_mask: "ee.Image",
    distance_pixels: int = 20,
    azimuth_width: int = 15,
) -> "ee.Image":
    """
    Create shadow mask using geometric projection from cloud positions.

    Projects cloud mask in the direction opposite to the sun, using
    directionalDistanceTransform. More accurate than spectral-only approach.

    Parameters:
    -----------
    image : ee.Image
        S2 SR image (needs MEAN_SOLAR_AZIMUTH_ANGLE property)
    cloud_mask : ee.Image
        Binary cloud mask (1=cloud, 0=clear); e.g. from cloud_probability_mask
    distance_pixels : int
        Maximum shadow projection distance in pixels (~10m each → × 100m)
    azimuth_width : int
        (Unused — kept for API compatibility) Angular width for shadow search

    Returns:
    --------
    ee.Image
        Single-band image 'shd_geo_mask' (1=shadow, 0=not shadow)
    """
    import ee

    # Shadow falls opposite to sun direction
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project cloud footprint in shadow direction at 100m for speed
    cld_proj = (
        cloud_mask
        .directionalDistanceTransform(shadow_azimuth, distance_pixels)
        .reproject(**{"crs": image.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("shd_geo_mask")
    )

    return cld_proj


def combine_masks(
    image: "ee.Image",
    use_cloud_prob: bool = True,
    use_shadow_spectral: bool = True,
    use_shadow_geometric: bool = False,
    buffer_meters: int = 50,
) -> "ee.Image":
    """
    Combine multiple masking methods into a single binary mask.

    Logic: OR operation — any method marking a pixel as bad → masked.

    Parameters:
    -----------
    image : ee.Image
        S2 SR image with 's2_cloudless' property
    use_cloud_prob : bool
        Include S2_CLOUD_PROBABILITY mask
    use_shadow_spectral : bool
        Include spectral (low-NIR) shadow mask
    use_shadow_geometric : bool
        Include geometric shadow projection (needs 'clouds' band from use_cloud_prob)
    buffer_meters : int
        Dilation buffer in meters applied after combining

    Returns:
    --------
    ee.Image
        Single-band image 'combined_mask' (1=masked, 0=valid)
    """
    import ee

    masks_to_combine = []

    if use_cloud_prob:
        cld = cloud_probability_mask(image, threshold=30)
        masks_to_combine.append(cld)

    if use_shadow_spectral:
        shd = shadow_mask_spectral(image, nir_threshold=0.15)
        masks_to_combine.append(shd)

    if use_shadow_geometric and use_cloud_prob:
        cld = masks_to_combine[0]  # Reuse cloud mask
        shd_geo = shadow_mask_geometric(image, cld, distance_pixels=20)
        masks_to_combine.append(shd_geo)

    # OR combination
    combined = masks_to_combine[0]
    for mask in masks_to_combine[1:]:
        combined = combined.add(mask)

    combined = combined.gt(0).rename("combined_mask")

    # Dilate: radius in 10m pixels = buffer_meters / 10
    if buffer_meters > 0:
        combined = combined.focalMax(buffer_meters / 10)

    return combined


def quality_assessment_band(image: "ee.Image") -> "ee.Image":
    """
    Create quality layer from Scene Classification Layer (SCL).

    SCL classes:
        0: No Data    1: Saturated/Defective   2: Dark Area
        3: Cloud Shadow   4: Vegetation   5: Not Vegetated
        6: Water   7: Unclassified   8: Cloud Medium Prob
        9: Cloud High Prob   10: Thin Cirrus   11: Snow/Ice

    Parameters:
    -----------
    image : ee.Image
        S2 image (must have SCL band)

    Returns:
    --------
    ee.Image
        Band 'qa_mask': 1=good pixel (vegetation/soil/water), 0=bad
    """
    import ee

    scl = image.select("SCL")
    good_pixels = scl.isin([4, 5, 6])  # vegetation, not-vegetated, water
    return good_pixels.rename("qa_mask")


def calculate_quality_score(
    image: "ee.Image",
    roi: "ee.Geometry",
) -> float:
    """
    Calculate overall data quality score for a single image over ROI.

    Based on combined mask: fraction of unmasked pixels in the ROI.

    Parameters:
    -----------
    image : ee.Image
        S2 image with 's2_cloudless' property
    roi : ee.Geometry
        Region of interest

    Returns:
    --------
    float
        Quality score [0, 1]: 1.0 = no clouds/shadows, 0.0 = fully masked

    Warning:
        Calls getInfo() — do NOT use inside ee.ImageCollection.map().
        For collection-wide ranking, use rank_scenes_by_quality() instead.
    """
    import ee

    mask = combine_masks(image)

    total = mask.reduceRegion(
        reducer=ee.Reducer.count(), geometry=roi, scale=10,
    ).getInfo()

    valid = mask.eq(0).reduceRegion(
        reducer=ee.Reducer.count(), geometry=roi, scale=10,
    ).getInfo()

    if not total or not valid:
        return 0.0

    total_val = list(total.values())[0] if total else 1
    valid_val = list(valid.values())[0] if valid else 0

    return float(max(0.0, min(1.0, valid_val / total_val))) if total_val > 0 else 0.0


def rank_scenes_by_quality(
    collection: "ee.ImageCollection",
    roi: "ee.Geometry",
    top_n: int = 10,
) -> list:
    """
    Rank all scenes in collection by data quality (descending).

    Uses CLOUDY_PIXEL_PERCENTAGE metadata (server-side, no per-image getInfo).
    Quality score = 1 − (CLOUDY_PIXEL_PERCENTAGE / 100).

    Parameters:
    -----------
    collection : ee.ImageCollection
        S2 image collection (must have CLOUDY_PIXEL_PERCENTAGE property)
    roi : ee.Geometry
        Region of interest (unused here, kept for API compatibility)
    top_n : int
        Return top N scenes

    Returns:
    --------
    list
        List of {date, quality} dicts, sorted by quality descending
    """
    import ee

    def add_quality_score(img):
        cloud_pct = ee.Number(img.get("CLOUDY_PIXEL_PERCENTAGE"))
        quality = ee.Number(1).subtract(cloud_pct.divide(100))
        return img.set("quality_score", quality)

    def to_feature(img):
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "quality": img.get("quality_score"),
        })

    fc = ee.FeatureCollection(collection.map(add_quality_score).map(to_feature))
    features = fc.getInfo()["features"]

    scenes = []
    for feat in features:
        props = feat["properties"]
        date = props.get("date")
        quality = props.get("quality")
        if date and quality is not None:
            scenes.append({"date": date, "quality": round(float(quality), 3)})

    scenes.sort(key=lambda x: x["quality"], reverse=True)
    return scenes[:top_n]


# =============================================================================
# Configuration presets
# =============================================================================


def preset_conservative() -> dict:
    """
    Conservative masking: aggressive cloud/shadow removal.
    Best for: High-accuracy requirements. Trade-off: more data loss.
    """
    return {
        "cloud_prob_threshold": 25,
        "nir_threshold": 0.20,
        "use_shadow_geometric": True,
        "buffer_meters": 100,
    }


def preset_balanced() -> dict:
    """
    Balanced masking: good trade-off between data loss and quality.
    Best for: Most applications (recommended).
    """
    return {
        "cloud_prob_threshold": 30,
        "nir_threshold": 0.15,
        "use_shadow_geometric": False,
        "buffer_meters": 50,
    }


def preset_aggressive() -> dict:
    """
    Aggressive masking: minimize data loss, accept some cloud contamination.
    Best for: Data-sparse regions. Trade-off: some cloudy pixels retained.
    """
    return {
        "cloud_prob_threshold": 40,
        "nir_threshold": 0.10,
        "use_shadow_geometric": False,
        "buffer_meters": 25,
    }
