"""Export Sentinel/Landsat drought-segmentation patches from Google Earth Engine.

This module is intentionally not run on import. Authenticate locally with
``earthengine authenticate`` before invoking the CLI or FastAPI export endpoint.
The output is educational research data only and is not a validated drought product.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests

if TYPE_CHECKING:
    from rasterio.enums import Resampling

from app.config import (
    DATASET_ROOT,
    GEE_GROWING_SEASON_END_MONTH,
    GEE_GROWING_SEASON_START_MONTH,
    PATCH_CHANNELS,
    PATCH_SCALE_METERS,
    PATCH_SIZE,
    VCI_DROUGHT_THRESHOLD_PERCENT,
)


# Approximate analysis windows for educational sampling, not administrative boundaries.
KAZAKHSTAN_REGIONS = {
    "akmola": [68.0, 50.5, 72.5, 54.5],
    "karaganda": [68.0, 45.0, 76.5, 50.8],
}
KAZAKHSTAN_CITIES = {
    "astana": (71.4304, 51.1694),
    "almaty": (76.8512, 43.2220),
    "shymkent": (69.5901, 42.3417),
    "karaganda_city": (73.1021, 49.8047),
    "aktobe": (57.1548, 50.2839),
    "atyrau": (51.9238, 47.0945),
    "kostanay": (63.6246, 53.2144),
    "kyzylorda": (65.5093, 44.8488),
    "pavlodar": (76.9674, 52.2873),
    "oskemen": (82.6285, 49.9483),
    "taraz": (71.3669, 42.9000),
    "turkistan": (68.2690, 43.3019),
    "semey": (80.2270, 50.4111),
    "oral": (51.3865, 51.2278),
    "petropavl": (69.1500, 54.8720),
}
SPLIT_RATIOS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
LANDSAT_CLEAR_PIXEL_BITS = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))


class EarthEngineConfigurationError(RuntimeError):
    """Raised when Earth Engine is unavailable or has not been authenticated."""


class SatelliteImageryUnavailableError(RuntimeError):
    """Raised when no imagery is available for the requested location and period."""


def scene_date_range(year: int, month: int | None = None) -> tuple[date, date]:
    """Return the half-open calendar window used for a coordinate composite."""
    if month is None:
        return date(year, GEE_GROWING_SEASON_START_MONTH, 1), date(year, GEE_GROWING_SEASON_END_MONTH + 1, 1)
    start = date(year, month, 1)
    end = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return start, end


def seasonal_date_range(year: int, period: str) -> tuple[date, date]:
    """Return the half-open satellite-composite window for a monitoring season."""
    if period == "warm":
        return date(year, 5, 1), date(year, 9, 1)
    if period == "cold":
        return date(year - 1, 11, 1), date(year, 4, 1)
    raise ValueError(f"Unknown monitoring period: {period}")


@dataclass(frozen=True)
class ExportResult:
    exported_patches: int
    dataset_root: Path


def _earth_engine():
    try:
        import ee
    except ImportError as error:  # pragma: no cover - dependency-level failure
        raise EarthEngineConfigurationError("earthengine-api is not installed.") from error
    return ee


def initialize_earth_engine() -> None:
    """Initialize a previously authenticated Earth Engine client without prompting.

    The server must use an existing Earth Engine credential or application-default
    credentials. This function never starts an interactive login flow.
    """
    ee = _earth_engine()
    try:
        project = os.getenv("GEE_PROJECT")
        service_account_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            if not project:
                raise ValueError("GEE_PROJECT must be set when using GEE_SERVICE_ACCOUNT_JSON.")
            key_data = json.loads(service_account_json)
            service_account = key_data.get("client_email")
            if not service_account:
                raise ValueError("The service-account JSON must contain client_email.")
            credentials = ee.ServiceAccountCredentials(service_account, key_data=service_account_json)
            ee.Initialize(credentials, project=project)
        elif project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as error:  # Earth Engine raises multiple implementation-specific errors.
        raise EarthEngineConfigurationError(
            "Google Earth Engine is not initialized. Set GEE_PROJECT and configure "
            "application-default credentials or the GEE_SERVICE_ACCOUNT_JSON Render secret."
        ) from error


def _add_ndvi(image):
    """Add a normalized-difference vegetation band to a harmonized image."""
    image = _earth_engine().Image(image)
    return image.addBands(image.normalizedDifference(["NIR", "Red"]).rename("NDVI"))


def mask_sentinel2(image):
    """Mask cloud/shadow/snow classes with Sentinel-2 L2A SCL."""
    image = _earth_engine().Image(image)
    scl = image.select("SCL")
    valid_surface = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    reflectance = image.select(
        ["B2", "B3", "B4", "B8", "B11", "B12"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    ).multiply(0.0001)
    return _add_ndvi(
        reflectance.updateMask(valid_surface).copyProperties(image, image.propertyNames())
    ).toFloat()


def mask_landsat(image):
    """Mask Landsat Collection 2 Level-2 cloud, shadow, snow, and fill pixels."""
    image = _earth_engine().Image(image)
    qa_pixel = image.select("QA_PIXEL")
    valid_surface = qa_pixel.bitwiseAnd(LANDSAT_CLEAR_PIXEL_BITS).eq(0)
    reflectance = image.select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    ).multiply(0.0000275).add(-0.2)
    return _add_ndvi(
        reflectance.updateMask(valid_surface).copyProperties(image, image.propertyNames())
    ).toFloat()


def build_composite_for_dates(region, scene_start: date, scene_end: date):
    """Return a seven-channel satellite composite and VCI drought label image."""
    ee = _earth_engine()
    start = ee.Date.fromYMD(scene_start.year, scene_start.month, scene_start.day)
    end = ee.Date.fromYMD(scene_end.year, scene_end.month, scene_end.day)
    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_sentinel2)
    )
    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_landsat)
    )
    observations = sentinel.merge(landsat).select(list(PATCH_CHANNELS))
    if observations.size().getInfo() == 0:
        raise SatelliteImageryUnavailableError(
            f"No usable satellite imagery found from {scene_start.isoformat()} through "
            f"{scene_end.isoformat()} at the requested location after cloud filtering."
        )
    composite = observations.median().select(list(PATCH_CHANNELS))
    ndvi_min = observations.select("NDVI").min()
    ndvi_max = observations.select("NDVI").max()
    ndvi_range = ndvi_max.subtract(ndvi_min).max(0.0001)
    vci = composite.select("NDVI").subtract(ndvi_min).divide(ndvi_range).multiply(100).rename("VCI")
    drought_mask = vci.lt(VCI_DROUGHT_THRESHOLD_PERCENT).rename("drought_mask").toUint8()
    return composite, drought_mask


def build_composite(region, year: int, month: int | None = None):
    """Return a calendar-month or default growing-season composite."""
    start, end = scene_date_range(year, month)
    return build_composite_for_dates(region, start, end)


def _grid_geometries(region, max_patches: int) -> Iterable:
    """Create deterministic, approximately 256 x 256 pixel grid cells in EPSG:3857."""
    ee = _earth_engine()
    patch_width_meters = PATCH_SIZE * PATCH_SCALE_METERS
    grid = region.coveringGrid(ee.Projection("EPSG:3857"), patch_width_meters).randomColumn("sample", 42).sort("sample")
    features = grid.toList(max_patches).getInfo()
    for feature in features:
        yield ee.Geometry(feature["geometry"])


def _read_raster(path: Path, bands: int, resampling: Resampling) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as source:
        if source.count != bands:
            raise ValueError(f"Expected {bands} bands in {path.name}, received {source.count}.")
        return source.read(
            out_shape=(bands, PATCH_SIZE, PATCH_SIZE),
            resampling=resampling,
        )


def _download_geotiff(image, path: Path, geometry) -> None:
    """Download one Earth Engine image and raise with the API's useful error text."""
    url = image.getDownloadURL(
        {
            "name": path.stem,
            "scale": PATCH_SCALE_METERS,
            "region": geometry,
            "crs": "EPSG:3857",
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    response = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=300)
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
            continue
        if response.status_code < 500 or attempt == 2:
            break
        time.sleep(2**attempt)
    if response is None:
        raise RuntimeError("Earth Engine tile download failed without a response.")
    if response.status_code != 200:
        try:
            detail = response.json()["error"]["message"]
        except (KeyError, ValueError, TypeError):
            detail = response.text[:500] or response.reason
        raise RuntimeError(f"Earth Engine tile download failed ({response.status_code}): {detail}")
    path.write_bytes(response.content)


def _download_patch(data_image, label_image, geometry, temp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download a single remote tile as GeoTIFFs and convert it to numpy arrays."""
    from rasterio.enums import Resampling

    patch_path = temp_dir / "patch.tif"
    stacked_image = data_image.addBands(label_image.toFloat())
    _download_geotiff(stacked_image, patch_path, geometry)
    patch = _read_raster(patch_path, len(PATCH_CHANNELS) + 1, Resampling.bilinear)
    image = patch[: len(PATCH_CHANNELS)].astype(np.float32)
    # Masked/no-data label pixels are not drought observations; make that policy
    # explicit before the integer cast to preserve the existing (label > 0) mask.
    label = np.nan_to_num(patch[len(PATCH_CHANNELS)], nan=0.0).astype(np.uint8)
    return np.nan_to_num(image, nan=0.0), (label > 0).astype(np.uint8)


def _split_name(random_source: random.Random) -> str:
    draw = random_source.random()
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS:
        cumulative += ratio
        if draw < cumulative:
            return name
    return "test"


def _city_split_map(cities: list[str]) -> dict[str, str]:
    """Keep all tiles from a city in one split to prevent nearby-patch leakage."""
    shuffled_cities = list(cities)
    random.Random(42).shuffle(shuffled_cities)
    train_count = int(len(shuffled_cities) * SPLIT_RATIOS[0][1])
    val_count = int(len(shuffled_cities) * SPLIT_RATIOS[1][1])
    return {
        city: split
        for split, split_cities in (
            ("train", shuffled_cities[:train_count]),
            ("val", shuffled_cities[train_count : train_count + val_count]),
            ("test", shuffled_cities[train_count + val_count :]),
        )
        for city in split_cities
    }


def _sampling_geometry(ee, region_name: str):
    if region_name in KAZAKHSTAN_REGIONS:
        return ee.Geometry.Rectangle(KAZAKHSTAN_REGIONS[region_name])
    if region_name in KAZAKHSTAN_CITIES:
        longitude, latitude = KAZAKHSTAN_CITIES[region_name]
        return ee.Geometry.Rectangle(
            [longitude - 0.5, latitude - 0.25, longitude + 0.5, latitude + 0.25]
        )
    raise ValueError(f"Unknown region or city: {region_name}")


def _write_patch(image: np.ndarray, label: np.ndarray, split: str, patch_id: str, dataset_root: Path) -> None:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_dir / f"{patch_id}.npy", image)
    np.save(label_dir / f"{patch_id}.npy", label)


def export_dataset(
    regions: list[str],
    year: int,
    max_patches: int,
    dataset_root: Path = DATASET_ROOT,
    progress_callback: Callable[[str], None] | None = None,
    split_by_region: bool = False,
) -> ExportResult:
    """Export GEE patches to train/val/test folders using a deterministic 70/15/15 split."""
    initialize_earth_engine()
    ee = _earth_engine()
    rng = random.Random(42)
    exported = 0
    processed = 0
    remaining = max_patches
    per_region_limit = max(1, (max_patches + len(regions) - 1) // len(regions))
    region_splits = _city_split_map(regions) if split_by_region else {}

    for region_name in regions:
        if progress_callback:
            progress_callback(f"Preparing {region_name} composite for {year}...")
        region = _sampling_geometry(ee, region_name)
        data_image, label_image = build_composite(region, year)
        for tile_number, geometry in enumerate(_grid_geometries(region, min(per_region_limit, remaining))):
            patch_id = f"{region_name}_{year}_{tile_number:05d}"
            split = region_splits.get(region_name)
            if split is None:
                split = _split_name(rng)
            image_path = dataset_root / split / "images" / f"{patch_id}.npy"
            label_path = dataset_root / split / "labels" / f"{patch_id}.npy"
            if image_path.is_file() and label_path.is_file():
                processed += 1
                exported += 1
                remaining -= 1
                if progress_callback:
                    progress_callback(
                        f"Processed {processed}/{max_patches} tiles; {exported} saved (already downloaded)."
                    )
                if remaining <= 0:
                    return ExportResult(exported_patches=exported, dataset_root=dataset_root)
                continue
            if progress_callback:
                progress_callback(
                    f"Downloading tile {processed + 1}/{max_patches} from {region_name}..."
                )
            with tempfile.TemporaryDirectory(prefix="gee_patch_") as temporary:
                image, label = _download_patch(data_image, label_image, geometry, Path(temporary))
            processed += 1
            if not np.isfinite(image).any() or image.max() == image.min():
                if progress_callback:
                    progress_callback(
                        f"Processed {processed}/{max_patches} tiles; {exported} saved (tile had no usable data)."
                    )
                continue
            _write_patch(image, label, split, patch_id, dataset_root)
            exported += 1
            remaining -= 1
            if progress_callback:
                progress_callback(f"Processed {processed}/{max_patches} tiles; {exported} saved.")
            if remaining <= 0:
                return ExportResult(exported_patches=exported, dataset_root=dataset_root)
    return ExportResult(exported_patches=exported, dataset_root=dataset_root)


def fetch_patch_for_coordinates(
    latitude: float, longitude: float, year: int, month: int | None = None
) -> np.ndarray:
    """Build and download a 256x256 seven-band patch centred on supplied coordinates."""
    initialize_earth_engine()
    ee = _earth_engine()
    point = ee.Geometry.Point([longitude, latitude])
    tile = point.buffer((PATCH_SIZE * PATCH_SCALE_METERS) / 2).bounds()
    data_image, label_image = build_composite(tile, year, month)
    with tempfile.TemporaryDirectory(prefix="gee_coordinate_") as temporary:
        image, _ = _download_patch(data_image, label_image, tile, Path(temporary))
    expected_shape = (len(PATCH_CHANNELS), PATCH_SIZE, PATCH_SIZE)
    if image.shape != expected_shape or not np.isfinite(image).all():
        raise ValueError(f"Earth Engine returned an invalid patch; expected shape {expected_shape}.")
    return image


def fetch_patch_for_period(latitude: float, longitude: float, year: int, period: str) -> np.ndarray:
    """Build and download a seven-band composite for a warm or cold season."""
    initialize_earth_engine()
    ee = _earth_engine()
    point = ee.Geometry.Point([longitude, latitude])
    tile = point.buffer((PATCH_SIZE * PATCH_SCALE_METERS) / 2).bounds()
    start, end = seasonal_date_range(year, period)
    data_image, label_image = build_composite_for_dates(tile, start, end)
    with tempfile.TemporaryDirectory(prefix="gee_seasonal_") as temporary:
        image, _ = _download_patch(data_image, label_image, tile, Path(temporary))
    expected_shape = (len(PATCH_CHANNELS), PATCH_SIZE, PATCH_SIZE)
    if image.shape != expected_shape or not np.isfinite(image).all():
        raise ValueError(f"Earth Engine returned an invalid seasonal patch; expected shape {expected_shape}.")
    return image
