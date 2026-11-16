import subprocess
import sys
from pathlib import Path


def test_app_startup_does_not_import_optional_heavy_libraries() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    script = """
import sys
import app.main
heavy_modules = {
    "torch", "torchvision", "segmentation_models_pytorch",
    "albumentations", "rasterio", "pandas", "sklearn",
}
loaded = heavy_modules.intersection(sys.modules)
assert not loaded, f"Heavy modules loaded during startup: {sorted(loaded)}"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=backend_dir,
        check=True,
        capture_output=True,
        text=True,
    )
