"""Export an evenly sampled city-surroundings dataset without model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import DATASET_ROOT
from ml.export_gee_dataset import KAZAKHSTAN_CITIES, export_dataset


DEFAULT_CITIES = [
    "astana",
    "almaty",
    "shymkent",
    "karaganda_city",
    "aktobe",
    "atyrau",
    "kostanay",
    "kyzylorda",
    "pavlodar",
    "oskemen",
    "taraz",
    "turkistan",
    "semey",
    "oral",
    "petropavl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--patches-per-city", type=int, default=3)
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES, choices=sorted(KAZAKHSTAN_CITIES))
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT.parent / "patches_kazakhstan_cities_2024",
    )
    arguments = parser.parse_args()
    if arguments.patches_per_city < 1:
        parser.error("--patches-per-city must be at least 1")
    if len(set(arguments.cities)) != len(arguments.cities):
        parser.error("--cities must not contain duplicates")

    total_patches = len(arguments.cities) * arguments.patches_per_city
    result = export_dataset(
        regions=arguments.cities,
        year=arguments.year,
        max_patches=total_patches,
        dataset_root=arguments.output,
        progress_callback=print,
        split_by_region=True,
    )
    print(
        f"Export complete: {result.exported_patches}/{total_patches} patches saved "
        f"to {result.dataset_root}."
    )


if __name__ == "__main__":
    main()
