"""Export labeled Earth Engine patches without training a model."""

from __future__ import annotations

import argparse

from ml.export_gee_dataset import export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-patches", type=int, default=20)
    parser.add_argument("--regions", nargs="+", default=["akmola", "karaganda"])
    arguments = parser.parse_args()

    result = export_dataset(
        regions=arguments.regions,
        year=arguments.year,
        max_patches=arguments.max_patches,
        progress_callback=print,
    )
    print(f"Export complete: {result.exported_patches} patches saved to {result.dataset_root}.")


if __name__ == "__main__":
    main()
