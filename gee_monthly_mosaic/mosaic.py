"""
Monthly cloud-free Sentinel-2 mosaic builder using Google Earth Engine.

Builds one median composite per calendar month for the last N months before
a reference date. The main output is a list of thumbnail URLs — one per month —
ready to be consumed by a frontend application.

Composites available (pass ``composite=`` to the constructor):
    "true_color"       B4, B3, B2  — default
    "false_color_nir"  B8, B4, B3  — vegetation in red
    "false_color_swir" B12, B8A, B4 — burn scars, soil moisture
    "agriculture"      B11, B8, B2  — crop/soil discrimination

Custom composites can be passed as a dict:
    composite={"bands": ["B8", "B4", "B3"], "min": 0, "max": 4000, "gamma": 1.4}

By default a 2–98% linear stretch is applied per band before generating the
thumbnail URL, which significantly improves contrast in low-dynamic-range areas.
Disable with ``stretch=False``.

Example:
    >>> import ee
    >>> ee.Initialize(project='mapbiomas')
    >>> geometry = ee.Geometry.Point(-57.87255, -10.03289).buffer(1000)
    >>> builder = MonthlyMosaicBuilder(geometry=geometry, end_date="2024-12-31")
    >>> results = builder.get_urls()
    [{"month": "2024-01", "url": "https://earthengine.googleapis.com/..."}, ...]

    >>> # Without stretch
    >>> builder = MonthlyMosaicBuilder(geometry, "2024-12-31", stretch=False)

    >>> # Export to Drive (optional)
    >>> task_ids = builder.export_to_drive(folder="mapbiomas-mosaics")
"""

import calendar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union

from gee_monthly_mosaic.acquisition import get_s2_sr_cld_col_masked

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cloud filtering constants
# ---------------------------------------------------------------------------
CLOUD_FILTER = 30
CLD_PRB_THRESH = 30
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

# ---------------------------------------------------------------------------
# Composite presets
# ---------------------------------------------------------------------------

COMPOSITES: Dict[str, dict] = {
    "true_color": {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "gamma": 1.4,
    },
    "false_color_nir": {
        "bands": ["B8", "B4", "B3"],
        "min": 0,
        "max": 4000,
        "gamma": 1.4,
    },
    "false_color_swir": {
        "bands": ["B12", "B8A", "B4"],
        "min": 0,
        "max": 4000,
        "gamma": 1.4,
    },
    "agriculture": {
        "bands": ["B11", "B8", "B2"],
        "min": 0,
        "max": 4000,
        "gamma": 1.4,
    },
}


# ---------------------------------------------------------------------------
# Geometry overlay
# ---------------------------------------------------------------------------

def paint_geometry(
    image: "ee.Image",
    geometry: "ee.Geometry",
    color: Union[int, str] = 0xFF0000,
    width: int = 2,
) -> "ee.Image":
    """
    Paint the geometry boundary onto an RGB image.

    Parameters
    ----------
    image : ee.Image
        3-band RGB image.
    geometry : ee.Geometry
        Geometry to paint (point, polygon, etc.).
    color : int or str
        Color as hex int (0xRRGGBB) or hex string ("FF0000" for red).
        Default: 0xFF0000 (red).
    width : int
        Stroke width in pixels (default: 2).

    Returns
    -------
    ee.Image
        RGB image with geometry painted as an overlay.
    """
    import ee

    if isinstance(color, str):
        color = int(color, 16)

    outline = ee.Image().byte().paint(geometry, color, width)
    return image.blend(outline)


# ---------------------------------------------------------------------------
# Linear stretch
# ---------------------------------------------------------------------------

def linear_stretch_2pct(image: "ee.Image", geometry: "ee.Geometry") -> "ee.Image":
    """
    Apply a 2–98% linear stretch per band to improve visual contrast.

    Computes the 2nd and 98th percentile of each band within ``geometry``
    in a single reduceRegion call, scales every pixel to [0, 255], and
    returns a byte image with bands named ["r", "g", "b"].

    Note: ee.Reducer.percentile([2, 98]) produces keys "{band}_p2" and
    "{band}_p98" — not "{band}".

    Parameters
    ----------
    image : ee.Image
        3-band image (already selected in RGB order).
    geometry : ee.Geometry
        Region used to compute the percentile statistics.

    Returns
    -------
    ee.Image
        3-band byte image with bands ["r", "g", "b"], values 0–255.
    """
    import ee

    img = image.rename(["r", "g", "b"])

    # getInfo() first — unitScale requires concrete floats, not ee.ComputedObject.
    # A null result means the mosaic has no valid pixels despite n_images > 0
    # (e.g. all pixels masked within the AOI). Fall back to fixed range in that case.
    stats = img.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=geometry,
        scale=30,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()

    # All-null stats → fully masked mosaic (no valid pixels in AOI this month)
    if all(stats.get(f"{b}_p2") is None for b in ["r", "g", "b"]):
        logger.warning("linear_stretch: all stats null — empty mosaic")
        return None

    def _stretch(band: str) -> "ee.Image":
        lo = stats.get(f"{band}_p2")
        hi = stats.get(f"{band}_p98")
        if lo is None or hi is None or lo == hi:
            lo, hi = 0.0, 3000.0
        return img.select(band).unitScale(lo, hi).clamp(0, 1).multiply(255).round().byte()

    r = _stretch("r")
    g = _stretch("g")
    b = _stretch("b")

    return ee.Image.cat(r, g, b).rename(["r", "g", "b"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_intervals(end_date: str, n_months: int) -> List[Tuple[str, str, str]]:
    """
    Return (label, start, end) tuples for the last n_months up to end_date.

    Each interval covers [first_day_of_month, first_day_of_next_month).

    Example:
        end_date="2024-12-15", n_months=3
        → [("2024-10", "2024-10-01", "2024-11-01"),
           ("2024-11", "2024-11-01", "2024-12-01"),
           ("2024-12", "2024-12-01", "2025-01-01")]
    """
    anchor = datetime.strptime(end_date, "%Y-%m-%d").replace(day=1)

    intervals = []
    for i in range(n_months - 1, -1, -1):
        year, month = anchor.year, anchor.month - i
        while month <= 0:
            month += 12
            year -= 1

        start = date(year, month, 1)
        next_month, next_year = month + 1, year
        if next_month > 12:
            next_month, next_year = 1, year + 1
        end = date(next_year, next_month, 1)

        intervals.append((
            f"{year:04d}-{month:02d}",
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        ))

    return intervals


def _resolve_composite(composite: Union[str, dict]) -> dict:
    if isinstance(composite, dict):
        return composite
    if composite not in COMPOSITES:
        raise ValueError(
            f"Unknown composite '{composite}'. "
            f"Available: {list(COMPOSITES)}. "
            "Pass a dict for a custom composite."
        )
    return COMPOSITES[composite]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MonthlyMosaicBuilder:
    """
    Cloud-free Sentinel-2 monthly mosaics for the last N months.

    The primary output is ``get_urls()``: a list of ``{month, url}`` dicts
    with GEE thumbnail URLs ready for frontend consumption.

    Parameters
    ----------
    geometry : ee.Geometry
        Area of interest.
    end_date : str
        Reference date "YYYY-MM-DD". Its month is the last in the series.
    n_months : int
        How many months back to go (default: 12).
    composite : str or dict
        Band composite to render. Either a preset name
        ("true_color", "false_color_nir", "false_color_swir", "agriculture")
        or a custom dict with keys ``bands``, ``min``, ``max``, ``gamma``.
        Default: "true_color".
    stretch : bool
        Apply 2–98% linear stretch per band before generating the URL.
        Greatly improves contrast. Default: True.
    paint_geometry : bool
        Paint the AOI geometry boundary onto the thumbnail. Default: True.
    geometry_color : str or int
        Color for the geometry outline (hex string "FF0000" or int 0xFF0000).
        Default: "FF0000" (red).
    geometry_width : int
        Width of the geometry stroke in pixels. Default: 2.
    dimensions : int
        Thumbnail size in pixels (longest side). Default: 512.
    cloud_filter : int
        Max cloud cover % for scene-level pre-filter (default: 30).
    cld_prb_thresh : int
        Cloud probability threshold for pixel masking (default: 30).
    nir_drk_thresh : float
        NIR darkness threshold for shadow detection (default: 0.15).
    cld_prj_dist : int
        Shadow projection distance ×100 m (default: 1).
    buffer : int
        Cloud mask dilation in meters (default: 50).
    """

    def __init__(
        self,
        geometry: "ee.Geometry",
        end_date: str,
        n_months: int = 12,
        composite: Union[str, dict] = "true_color",
        stretch: bool = True,
        paint_geometry: bool = True,
        geometry_color: Union[str, int] = "FF0000",
        geometry_width: int = 2,
        dimensions: int = 512,
        cloud_filter: int = CLOUD_FILTER,
        cld_prb_thresh: int = CLD_PRB_THRESH,
        nir_drk_thresh: float = NIR_DRK_THRESH,
        cld_prj_dist: int = CLD_PRJ_DIST,
        buffer: int = BUFFER,
    ) -> None:
        self.geometry = geometry
        self.end_date = end_date
        self.n_months = n_months
        self.vis = _resolve_composite(composite)
        self.stretch = stretch
        self.paint_geometry = paint_geometry
        self.geometry_color = geometry_color
        self.geometry_width = geometry_width
        self.dimensions = dimensions
        self.cloud_filter = cloud_filter
        self.cld_prb_thresh = cld_prb_thresh
        self.nir_drk_thresh = nir_drk_thresh
        self.cld_prj_dist = cld_prj_dist
        self.buffer = buffer

        self._mosaics: Optional[Dict[str, "ee.Image"]] = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_mosaic(self, label: str, start: str, end: str) -> "ee.Image":
        """Return a median mosaic ee.Image for the given interval (lazy, no getInfo)."""
        import ee

        col = get_s2_sr_cld_col_masked(
            aoi=self.geometry,
            start_date=start,
            end_date=end,
            cloud_filter=self.cloud_filter,
            cld_prb_thresh=self.cld_prb_thresh,
            nir_drk_thresh=self.nir_drk_thresh,
            cld_prj_dist=self.cld_prj_dist,
            buffer=self.buffer,
        )
        return col.median().set("system:time_start", ee.Date(start).millis())

    def _count_images_per_month(self) -> Dict[str, int]:
        """
        Count the number of valid Sentinel-2 images per month (batched).

        Makes a single getInfo() call for all months combined, returning a dict
        {label: count}. Months with 0 images are included.
        """
        import ee

        counts_by_month = {}
        counts_dict = {}

        for label, start, end in _month_intervals(self.end_date, self.n_months):
            col = get_s2_sr_cld_col_masked(
                aoi=self.geometry,
                start_date=start,
                end_date=end,
                cloud_filter=self.cloud_filter,
                cld_prb_thresh=self.cld_prb_thresh,
                nir_drk_thresh=self.nir_drk_thresh,
                cld_prj_dist=self.cld_prj_dist,
                buffer=self.buffer,
            )
            counts_dict[label] = col.size()

        # Single batched getInfo() for all months
        counts_list = ee.List(list(counts_dict.values())).getInfo()
        for label, count in zip(counts_dict.keys(), counts_list):
            counts_by_month[label] = count

        return counts_by_month

    def _url_for_month(
        self, label: str, start: str, end: str, region: list, n_images: int
    ) -> Dict[str, any]:
        """Build and return {month, url, n_images} for a single month. Runs in a thread."""
        mosaic = self._build_mosaic(label, start, end)
        img_rgb = mosaic.select(self.vis["bands"])

        if self.stretch:
            img_rgb = linear_stretch_2pct(img_rgb, self.geometry)
            # All stats null → empty mosaic (no valid pixels in AOI this month)
            if img_rgb is None:
                return {"month": label, "url": None, "n_images": n_images}
            thumb_params = {
                "bands": ["r", "g", "b"],
                "min": 0,
                "max": 255,
                "dimensions": self.dimensions,
                "region": region,
                "format": "png",
            }
        else:
            thumb_params = {
                "min": self.vis["min"],
                "max": self.vis["max"],
                "gamma": self.vis.get("gamma", 1.0),
                "dimensions": self.dimensions,
                "region": region,
                "format": "png",
            }

        if self.paint_geometry:
            img_rgb = paint_geometry(img_rgb, self.geometry, self.geometry_color, self.geometry_width)

        url = img_rgb.getThumbURL(thumb_params)
        logger.info(f"[{label}] URL generated ({n_images} images)")
        return {"month": label, "url": url, "n_images": n_images}

    def _build_all(self) -> Dict[str, "ee.Image"]:
        if self._mosaics is None:
            self._mosaics = {}
            for label, start, end in _month_intervals(self.end_date, self.n_months):
                mosaic, _ = self._build_mosaic(label, start, end)
                self._mosaics[label] = mosaic
        return self._mosaics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_urls(self, workers: int = 6) -> List[Dict[str, any]]:
        """
        Generate thumbnail URLs for all monthly mosaics in parallel.

        Each month is processed in its own thread — building the median
        composite, computing the linear stretch stats, and calling
        getThumbURL are all independent across months.

        Image counts are fetched in a single batched getInfo() call before
        parallelization starts.

        URLs expire after ~2 hours (standard GEE behaviour).
        Months with no valid images return ``url: null``.

        Parameters
        ----------
        workers : int
            Number of parallel threads (default: 6). Increasing beyond 8–10
            rarely helps due to GEE server-side rate limits.

        Returns
        -------
        list of dict
            [{"month": "YYYY-MM", "url": "https://...", "n_images": 5}, ...]
            Ordered chronologically. ``url`` is None for empty months,
            ``n_images`` is the count of valid Sentinel-2 images in the median.
        """
        logger.info("Counting images per month (batched)...")
        image_counts = self._count_images_per_month()

        region = self.geometry.buffer(500).bounds().getInfo()["coordinates"]
        intervals = _month_intervals(self.end_date, self.n_months)

        results: Dict[str, Dict] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._url_for_month, label, start, end, region, image_counts[label]
                ): label
                for label, start, end in intervals
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result()
                except Exception as exc:
                    logger.error(f"[{label}] failed: {exc}")
                    results[label] = {"month": label, "url": None, "n_images": 0}

        return [results[label] for label, _, _ in intervals]

    def export_to_drive(
        self,
        folder: str = "ee-mapbiomas",
        scale: int = 30,
        crs: str = "EPSG:4326",
    ) -> List[str]:
        """
        Export each monthly mosaic (all bands) to Google Drive.

        Submits GEE export tasks asynchronously. Monitor in the GEE Tasks panel.

        Returns
        -------
        list of str
            Task IDs, one per month.
        """
        import ee

        task_ids = []
        for label, mosaic in self._build_all().items():
            filename = f"s2_mosaic_{label}"
            task = ee.batch.Export.image.toDrive(
                image=mosaic.clip(self.geometry),
                description=filename,
                folder=folder,
                fileNamePrefix=filename,
                scale=scale,
                crs=crs,
                maxPixels=1e13,
            )
            task.start()
            task_ids.append(task.id)
            logger.info(f"[{label}] Drive export started (task={task.id})")

        return task_ids

    def export_to_gcs(
        self,
        bucket: str,
        folder: str = "mosaics",
        scale: int = 30,
        crs: str = "EPSG:4326",
    ) -> List[str]:
        """
        Export each monthly mosaic (all bands) to Google Cloud Storage.

        Returns
        -------
        list of str
            Task IDs, one per month.
        """
        import ee

        task_ids = []
        for label, mosaic in self._build_all().items():
            filename = f"s2_mosaic_{label}"
            task = ee.batch.Export.image.toCloudStorage(
                image=mosaic.clip(self.geometry),
                description=filename,
                bucket=bucket,
                fileNamePrefix=f"{folder}/{filename}",
                scale=scale,
                crs=crs,
                maxPixels=1e13,
            )
            task.start()
            task_ids.append(task.id)
            logger.info(f"[{label}] GCS export started (task={task.id})")

        return task_ids
