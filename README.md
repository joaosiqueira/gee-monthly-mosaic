# GEE Monthly Mosaic

Cloud-free Sentinel-2 monthly mosaics via Google Earth Engine.

Generates one median composite per calendar month for the last N months before a reference date, with automatic cloud/shadow masking and optional linear stretch for improved contrast.

**Primary output:** a list of thumbnail URLs ready for frontend consumption.

```python
[
    {"month": "2024-01", "url": "https://...", "n_images": 5},
    {"month": "2024-02", "url": "https://...", "n_images": 12},
    ...
]
```

---

## Installation

```bash
pip install gee-monthly-mosaic
```

With notebook support (for visualization):
```bash
pip install gee-monthly-mosaic[notebook]
```

---

## Quick Start

```python
import ee
from gee_monthly_mosaic import MonthlyMosaicBuilder

# Initialize Google Earth Engine
ee.Initialize(project='your-gee-project')

# Define an area of interest
geometry = ee.Geometry.Point(-57.87255, -10.03289).buffer(1000)

# Build monthly mosaics
builder = MonthlyMosaicBuilder(
    geometry=geometry,
    end_date="2024-12-31",
    n_months=12,
)

# Get thumbnail URLs (parallel processing, ~6 workers)
results = builder.get_urls()

# Each result:
# {"month": "2024-01", "url": "https://earthengine.googleapis.com/...", "n_images": 5}
for r in results:
    print(f"{r['month']}: {r['n_images']} images → {r['url'][:50]}...")
```

---

## Features

### Band Composites

Preset band combinations (set `composite=` parameter):

| Name | Bands | Use Case |
|------|-------|----------|
| `true_color` | B4, B3, B2 | Natural color (default) |
| `false_color_nir` | B8, B4, B3 | Vegetation in red — deforestation detection |
| `false_color_swir` | B12, B8A, B4 | Burn scars, soil moisture |
| `agriculture` | B11, B8, B2 | Crop/soil discrimination |

Or pass a custom dict:
```python
custom = {"bands": ["B8", "B4", "B3"], "min": 0, "max": 5000, "gamma": 1.8}
builder = MonthlyMosaicBuilder(geometry=geometry, composite=custom)
```

### Linear Stretch (2–98%)

Automatically applied by default. Computes the 2nd and 98th percentile of each band within the geometry and scales pixel values to [0, 255] — dramatically improves contrast.

Disable with `stretch=False`:
```python
builder = MonthlyMosaicBuilder(geometry=geometry, stretch=False)
```

### Geometry Overlay

Paints the AOI boundary as a colored outline on each thumbnail.

```python
builder = MonthlyMosaicBuilder(
    geometry=geometry,
    paint_geometry=True,           # enable (default)
    geometry_color="FF0000",       # red hex (default)
    geometry_width=2,              # pixels
)
```

### Cloud Masking

Configurable cloud and shadow detection (default thresholds are conservative):

```python
builder = MonthlyMosaicBuilder(
    geometry=geometry,
    cloud_filter=30,               # % max cloud cover (scene-level pre-filter)
    cld_prb_thresh=30,             # cloud probability threshold (0–100)
    nir_drk_thresh=0.15,           # NIR darkness threshold for shadows
    cld_prj_dist=1,                # shadow projection distance (×100 m)
    buffer=50,                     # cloud mask dilation (meters)
)
```

### Parallel Processing

URLs are generated in parallel across months. Default 6 workers; increase for faster processing (GEE limits around 8–10 due to rate limits):

```python
results = builder.get_urls(workers=12)
```

---

## Output Format

Each result is a dict with three keys:

```python
{
    "month": "2024-01",                    # ISO year-month
    "url": "https://earthengine.googleapis.com/...",  # expires ~2 hours
    "n_images": 5,                         # number of images in median composite
}
```

Months with no valid imagery (e.g., persistent cloud cover) have `"url": None`.

---

## Export to Drive / GCS (Optional)

Export full-resolution mosaics (all bands) for further processing:

```python
# Google Drive
task_ids = builder.export_to_drive(folder="mapbiomas-mosaics", scale=30)

# Google Cloud Storage
task_ids = builder.export_to_gcs(bucket="my-bucket", folder="mosaics", scale=30)
```

Monitor progress in the GEE Tasks panel.

---

## Cloud Masking Details

The pipeline chains five steps:

1. **Cloud probability** — S2_CLOUD_PROBABILITY ≥ threshold → masked
2. **Shadow detection (spectral)** — low NIR ≤ threshold → potential shadow
3. **Shadow detection (geometric)** — project clouds opposite to sun direction
4. **Combine** — OR-combine all masks, remove small patches, dilate
5. **Apply mask** — mask all reflectance bands

Inspired by the [GEE Cloud Masking tutorial](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-cloud-masking).

---

## API Reference

### `MonthlyMosaicBuilder`

Main class. Constructor and key methods:

**`__init__(...)`**
- `geometry` — ee.Geometry (point, polygon, etc.)
- `end_date` — str "YYYY-MM-DD"; last month is the month containing this date
- `n_months` — int (default: 12)
- `composite` — str (preset name) or dict (custom vis params); default: "true_color"
- `stretch` — bool (apply 2–98% linear stretch); default: True
- `paint_geometry` — bool (overlay AOI boundary); default: True
- `geometry_color` — str or int (hex color); default: "FF0000"
- `geometry_width` — int (pixels); default: 2
- `dimensions` — int (thumbnail size, longest side); default: 512
- Cloud masking params: `cloud_filter`, `cld_prb_thresh`, `nir_drk_thresh`, `cld_prj_dist`, `buffer`

**`get_urls(workers=6)`** → `List[Dict[str, any]]`
- Generate thumbnail URLs in parallel
- Returns list of `{"month": str, "url": str|None, "n_images": int}`

**`export_to_drive(...)`** → `List[str]`
- Export full-resolution mosaics to Google Drive
- Returns task IDs

**`export_to_gcs(...)`** → `List[str]`
- Export to Google Cloud Storage
- Returns task IDs

### `get_s2_sr_cld_col_masked(...)`

Lower-level function: returns an `ee.ImageCollection` of cloud-masked Sentinel-2 scenes for a given date range and AOI.

```python
from gee_monthly_mosaic import get_s2_sr_cld_col_masked

col = get_s2_sr_cld_col_masked(
    aoi=geometry,
    start_date="2024-01-01",
    end_date="2024-12-31",
    cloud_filter=30,
    cld_prb_thresh=30,
    # ... other masking params
)
```

### `extract_ndvi_series(...)`

Extract NDVI time series from a masked collection:

```python
from gee_monthly_mosaic import extract_ndvi_series

series = extract_ndvi_series(col, geometry)
# [{"date": "2024-01-01", "ndvi": 0.82, "valid_pct": 0.95}, ...]
```

---

## Example: Compare Stretch vs. No Stretch

```python
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

months = ["2024-08", "2024-09"]
composites = ["true_color", "false_color_nir"]

fig, axes = plt.subplots(len(months), len(composites) * 2, figsize=(16, 8))

for row, month_end in enumerate(months):
    for col, comp_name in enumerate(composites):
        # With stretch
        builder_stretch = MonthlyMosaicBuilder(
            geometry=geometry,
            end_date=month_end,
            n_months=1,
            composite=comp_name,
            stretch=True,
        )
        urls_stretch = builder_stretch.get_urls(workers=1)

        # Without stretch
        builder_no_stretch = MonthlyMosaicBuilder(
            geometry=geometry,
            end_date=month_end,
            n_months=1,
            composite=comp_name,
            stretch=False,
        )
        urls_no_stretch = builder_no_stretch.get_urls(workers=1)

        # Plot
        for url_idx, url_dict in enumerate([urls_stretch[0], urls_no_stretch[0]]):
            if url_dict["url"]:
                resp = requests.get(url_dict["url"])
                img = Image.open(BytesIO(resp.content))
                ax = axes[row][col * 2 + url_idx]
                ax.imshow(img)
                ax.set_title(f"{comp_name}\n{'WITH' if url_idx == 0 else 'NO'} stretch")
                ax.axis("off")

plt.tight_layout()
plt.show()
```

---

## Requirements

- Python ≥ 3.8
- `earthengine-api` ≥ 0.1.300 — must have `ee.Authenticate()` run once in your environment

Optional: for notebook visualization, install with `[notebook]` extras.

---

## License

MIT

---

## Author

MapBiomas
