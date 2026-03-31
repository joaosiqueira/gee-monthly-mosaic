"""
GEE Monthly Mosaic — Cloud-free Sentinel-2 monthly mosaics via Google Earth Engine.

Main API:
    MonthlyMosaicBuilder — generates thumbnail URLs for 12 monthly mosaics
    COMPOSITES — preset band combinations (true_color, false_color_nir, etc.)
    linear_stretch_2pct — 2–98% linear stretch per band
    paint_geometry — overlays AOI boundary on RGB image

Cloud masking:
    get_s2_sr_cld_col_masked — full Sentinel-2 SR pipeline with cloud/shadow masking
    extract_ndvi_series — NDVI extraction with valid pixel percentages

Example:
    >>> import ee
    >>> from gee_monthly_mosaic import MonthlyMosaicBuilder
    >>> ee.Initialize(project='mapbiomas')
    >>> geometry = ee.Geometry.Point(-57.87, -10.03).buffer(1000)
    >>> builder = MonthlyMosaicBuilder(geometry=geometry, end_date="2024-12-31")
    >>> results = builder.get_urls()  # [{"month": "2024-01", "url": "...", "n_images": 5}, ...]
"""

__version__ = "0.1.0"

from gee_monthly_mosaic.mosaic import (
    MonthlyMosaicBuilder,
    COMPOSITES,
    linear_stretch_2pct,
    paint_geometry,
)
from gee_monthly_mosaic.acquisition import (
    get_s2_sr_cld_col_masked,
    extract_ndvi_series,
    get_ndvi_series_for_aoi,
)

__all__ = [
    "MonthlyMosaicBuilder",
    "COMPOSITES",
    "linear_stretch_2pct",
    "paint_geometry",
    "get_s2_sr_cld_col_masked",
    "extract_ndvi_series",
    "get_ndvi_series_for_aoi",
]
