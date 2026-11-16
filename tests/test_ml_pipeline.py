from io import BytesIO
import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi.testclient import TestClient

from app.api import routes_segmentation
from app.config import PATCH_CHANNELS, PATCH_SIZE
from app.main import app
from app.services.segmentation_service import SegmentationModelUnavailableError
from ml.dataset import NpyDroughtPatchDataset, normalize_patch
from ml.export_gee_dataset import (
    EarthEngineConfigurationError,
    _city_split_map,
    _download_geotiff,
    _download_patch,
    export_dataset,
    initialize_earth_engine,
)
from ml.train import TrainingConfig, train_segmentation


def test_normalize_patch_has_seven_channels_and_safe_range() -> None:
    patch = np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    patch[6] = -1.0
    normalized = normalize_patch(patch)
    assert normalized.shape == patch.shape
    assert normalized.min() >= 0
    assert normalized.max() <= 1


def test_initialize_earth_engine_uses_service_account_environment_secret(monkeypatch) -> None:
    key_json = json.dumps({"client_email": "aqua-demo@example.iam.gserviceaccount.com", "private_key": "test-key"})
    calls = {}
    credentials = object()

    def create_credentials(email, key_data):
        calls["credentials"] = (email, key_data)
        return credentials

    def initialize(actual_credentials, project):
        calls["initialize"] = (actual_credentials, project)

    monkeypatch.setenv("GEE_PROJECT", "aqua-demo-project")
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_JSON", key_json)
    monkeypatch.setitem(
        sys.modules,
        "ee",
        SimpleNamespace(ServiceAccountCredentials=create_credentials, Initialize=initialize),
    )

    initialize_earth_engine()

    assert calls["credentials"] == ("aqua-demo@example.iam.gserviceaccount.com", key_json)
    assert calls["initialize"] == (credentials, "aqua-demo-project")


def test_initialize_earth_engine_requires_project_for_service_account_secret(monkeypatch) -> None:
    key_json = json.dumps({"client_email": "aqua-demo@example.iam.gserviceaccount.com"})
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_JSON", key_json)
    monkeypatch.setitem(sys.modules, "ee", SimpleNamespace())

    try:
        initialize_earth_engine()
    except EarthEngineConfigurationError as error:
        assert "GEE_SERVICE_ACCOUNT_JSON" in str(error)
    else:
        raise AssertionError("Expected missing project configuration to fail explicitly")


def test_npy_dataset_loads_paired_patch(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    np.save(images / "sample.npy", np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
    np.save(labels / "sample.npy", np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8))
    dataset = NpyDroughtPatchDataset(images, labels)
    image, label = dataset[0]
    assert tuple(image.shape) == (7, PATCH_SIZE, PATCH_SIZE)
    assert tuple(label.shape) == (1, PATCH_SIZE, PATCH_SIZE)


def test_fine_tuning_starts_from_checkpoint_and_preserves_best_baseline(monkeypatch, tmp_path) -> None:
    class TinySegmenter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(len(PATCH_CHANNELS), 1, kernel_size=1)

        def forward(self, image):
            return self.projection(image)

    images = torch.zeros((2, len(PATCH_CHANNELS), 4, 4))
    labels = torch.zeros((2, 1, 4, 4))
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    monkeypatch.setattr("ml.train.create_dataloaders", lambda **_kwargs: (loader, loader, loader))
    monkeypatch.setattr("ml.train.build_unet", lambda _encoder: TinySegmenter())
    initial_path = tmp_path / "initial.pth"
    candidate_path = tmp_path / "candidate.pth"
    initial_model = TinySegmenter()
    torch.save(
        {
            "state_dict": initial_model.state_dict(),
            "encoder_name": "resnet34",
            "in_channels": len(PATCH_CHANNELS),
            "epoch": 7,
            "validation_loss": 1.0,
        },
        initial_path,
    )

    result = train_segmentation(
        TrainingConfig(epochs=1, learning_rate=1e-4),
        dataset_root=tmp_path,
        checkpoint_path=candidate_path,
        initial_checkpoint_path=initial_path,
    )

    saved = torch.load(candidate_path, map_location="cpu", weights_only=True)
    assert np.isfinite(result["initial_validation_loss"])
    assert result["best_validation_loss"] <= result["initial_validation_loss"]
    assert saved["fine_tuned_from"] == str(initial_path)
    assert saved["in_channels"] == len(PATCH_CHANNELS)
    assert saved["epoch"] >= 7


def test_uploaded_tile_returns_clear_model_not_ready_response(monkeypatch) -> None:
    def unavailable_model(*_args, **_kwargs):
        raise SegmentationModelUnavailableError("Segmentation model is unavailable for this test.")

    monkeypatch.setattr(routes_segmentation, "infer_patch", unavailable_model)
    buffer = BytesIO()
    np.save(buffer, np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
    client = TestClient(app)
    response = client.post(
        "/api/ml/infer/upload",
        files={"tile": ("tile.npy", buffer.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 409
    assert "Segmentation model is unavailable" in response.json()["detail"]


def test_download_patch_maps_nan_labels_to_non_drought_without_warning(monkeypatch, tmp_path) -> None:
    class FakeImage:
        def toFloat(self):
            return self

        def addBands(self, _label):
            return self

    patch = np.zeros((8, 2, 2), dtype=np.float32)
    patch[6] = np.nan
    patch[7] = np.array([[0.0, 1.0], [2.0, np.nan]])
    monkeypatch.setattr("ml.export_gee_dataset._download_geotiff", lambda *_args: None)
    monkeypatch.setattr("ml.export_gee_dataset._read_raster", lambda *_args: patch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        image, label = _download_patch(FakeImage(), FakeImage(), FakeImage(), tmp_path)
    assert not caught
    assert image.shape == (7, 2, 2)
    assert label.tolist() == [[0, 1], [1, 0]]


def test_export_dataset_reports_patch_download_progress(monkeypatch, tmp_path) -> None:
    class FakeGeometry:
        @staticmethod
        def Rectangle(_bounds):
            return object()

    class FakeEarthEngine:
        Geometry = FakeGeometry

    monkeypatch.setattr("ml.export_gee_dataset.initialize_earth_engine", lambda: None)
    monkeypatch.setattr("ml.export_gee_dataset._earth_engine", lambda: FakeEarthEngine)
    monkeypatch.setattr("ml.export_gee_dataset.build_composite", lambda *_args: (object(), object()))
    monkeypatch.setattr("ml.export_gee_dataset._grid_geometries", lambda *_args: iter([object()]))
    monkeypatch.setattr(
        "ml.export_gee_dataset._download_patch",
        lambda *_args: (np.array([[[0.0, 1.0]]]), np.zeros((1, 2), dtype=np.uint8)),
    )

    messages = []
    result = export_dataset(
        ["akmola"],
        2024,
        1,
        dataset_root=tmp_path,
        progress_callback=messages.append,
    )

    assert result.exported_patches == 1
    assert messages == [
        "Preparing akmola composite for 2024...",
        "Downloading tile 1/1 from akmola...",
        "Processed 1/1 tiles; 1 saved.",
    ]


def test_city_splits_keep_cities_disjoint() -> None:
    cities = [f"city_{index}" for index in range(15)]
    split_map = _city_split_map(cities)
    assert {split: list(split_map.values()).count(split) for split in ("train", "val", "test")} == {
        "train": 10,
        "val": 2,
        "test": 3,
    }
    assert set(split_map) == set(cities)


def test_city_sampling_geometry_uses_area_around_city(monkeypatch) -> None:
    from ml.export_gee_dataset import _sampling_geometry

    captured = []

    class FakeGeometry:
        @staticmethod
        def Rectangle(bounds):
            captured.extend(bounds)
            return bounds

    class FakeEarthEngine:
        Geometry = FakeGeometry

    bounds = _sampling_geometry(FakeEarthEngine, "astana")
    assert len(bounds) == 4
    assert bounds[0] < bounds[2]
    assert bounds[1] < bounds[3]
    assert captured == bounds


def test_export_dataset_skips_already_downloaded_city_patch(monkeypatch, tmp_path) -> None:
    from ml.export_gee_dataset import _city_split_map

    class FakeGeometry:
        @staticmethod
        def Rectangle(_bounds):
            return object()

    class FakeEarthEngine:
        Geometry = FakeGeometry

    split = _city_split_map(["astana"])["astana"]
    image_path = tmp_path / split / "images" / "astana_2024_00000.npy"
    label_path = tmp_path / split / "labels" / "astana_2024_00000.npy"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.touch()
    label_path.touch()

    monkeypatch.setattr("ml.export_gee_dataset.initialize_earth_engine", lambda: None)
    monkeypatch.setattr("ml.export_gee_dataset._earth_engine", lambda: FakeEarthEngine)
    monkeypatch.setattr("ml.export_gee_dataset.build_composite", lambda *_args: (object(), object()))
    monkeypatch.setattr("ml.export_gee_dataset._grid_geometries", lambda *_args: iter([object()]))
    monkeypatch.setattr(
        "ml.export_gee_dataset._download_patch",
        lambda *_args: (_ for _ in ()).throw(AssertionError("existing patch must not be downloaded")),
    )

    messages = []
    result = export_dataset(
        ["astana"],
        2024,
        1,
        dataset_root=tmp_path,
        progress_callback=messages.append,
        split_by_region=True,
    )

    assert result.exported_patches == 1
    assert messages[-1] == "Processed 1/1 tiles; 1 saved (already downloaded)."


def test_download_geotiff_retries_transient_connection_error(monkeypatch, tmp_path) -> None:
    class FakeImage:
        @staticmethod
        def getDownloadURL(_params):
            return "https://example.test/tile"

    class FakeResponse:
        status_code = 200
        content = b"geotiff"

    responses = iter(
        [
            requests.ConnectionError("transient"),
            FakeResponse(),
        ]
    )
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("ml.export_gee_dataset.requests.get", fake_get)
    monkeypatch.setattr("ml.export_gee_dataset.time.sleep", lambda _seconds: None)
    output_path = tmp_path / "patch.tif"

    _download_geotiff(FakeImage(), output_path, object())

    assert len(calls) == 2
    assert output_path.read_bytes() == b"geotiff"
