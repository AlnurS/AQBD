"""Create a database snapshot containing only national screening-grid data."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


BASELINE_CITIES = (
    ("Алматы", 43.2389, 76.8897),
    ("Астана", 51.1694, 71.4491),
    ("Шымкент", 42.3417, 69.5901),
    ("Караганда", 49.8047, 73.1094),
    ("Актобе", 50.2839, 57.1670),
    ("Тараз", 42.9000, 71.3667),
    ("Павлодар", 52.2873, 76.9674),
    ("Усть-Каменогорск", 49.9483, 82.6285),
    ("Семей", 50.4111, 80.2275),
    ("Атырау", 47.0945, 51.9238),
    ("Костанай", 53.2144, 63.6246),
    ("Кызылорда", 44.8488, 65.4823),
    ("Петропавловск", 54.8720, 69.1500),
    ("Туркестан", 43.3019, 68.2690),
    ("Уральск", 51.2278, 51.3865),
    ("Актау", 43.6532, 51.1975),
    ("Кокшетау", 53.2833, 69.3833),
    ("Талдыкорган", 45.0156, 78.3739),
    ("Темиртау", 50.0549, 72.9646),
    ("Экибастуз", 51.7237, 75.3229),
    ("Рудный", 52.9529, 63.1308),
)


def add_baseline_city_locations(connection: sqlite3.Connection) -> int:
    """Insert stable, selectable city monitoring points into the public snapshot."""
    created_at = datetime.now(timezone.utc).isoformat()
    before = connection.total_changes
    connection.executemany(
        """INSERT OR IGNORE INTO monitoring_locations
           (id, name, latitude, longitude, region, created_at, active, location_type)
           VALUES (?, ?, ?, ?, ?, ?, 1, 'user_point')""",
        [
            (
                str(uuid5(NAMESPACE_URL, f"https://aqua-kz.example/city/{name}")),
                name,
                latitude,
                longitude,
                "Базовая точка по Казахстану",
                created_at,
            )
            for name, latitude, longitude in BASELINE_CITIES
        ],
    )
    return connection.total_changes - before


def create_public_seed(source_path: Path, output_path: Path) -> tuple[int, int, int]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("The public seed output must not overwrite the source database.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{source_path.as_uri()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(output_path) as output:
        required_tables = {"monitoring_locations", "monitoring_observations", "national_grid_cells"}
        available_tables = {
            row[0]
            for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing_tables = required_tables - available_tables
        if missing_tables:
            raise ValueError(f"Source database is missing tables: {', '.join(sorted(missing_tables))}")

        source.backup(output)
        output.execute("PRAGMA foreign_keys = OFF")
        output.execute(
            """DELETE FROM monitoring_observations
               WHERE location_id NOT IN (SELECT location_id FROM national_grid_cells)"""
        )
        output.execute(
            """DELETE FROM monitoring_locations
               WHERE id NOT IN (SELECT location_id FROM national_grid_cells)"""
        )
        if "inference_runs" in available_tables:
            output.execute(
                """DELETE FROM inference_runs
                   WHERE id NOT IN (SELECT inference_run_id FROM monitoring_observations)"""
            )
            output.execute(
                """UPDATE inference_runs
                   SET upload_filename = NULL,
                       mask_path = '/app/data/predictions/' || id || '_mask.npy'"""
            )
        if "national_grid_status" in available_tables:
            output.execute(
                """DELETE FROM national_grid_status
                   WHERE NOT EXISTS (
                       SELECT 1 FROM national_grid_cells c
                       WHERE c.grid_version = national_grid_status.grid_version
                         AND c.cell_id = national_grid_status.cell_id
                   )"""
            )
        add_baseline_city_locations(output)
        output.commit()

        location_count = output.execute("SELECT COUNT(*) FROM monitoring_locations").fetchone()[0]
        grid_count = output.execute("SELECT COUNT(*) FROM national_grid_cells").fetchone()[0]
        city_count = output.execute(
            """SELECT COUNT(*) FROM monitoring_locations
               WHERE location_type = 'user_point'
                 AND region = 'Базовая точка по Казахстану'"""
        ).fetchone()[0]
        observation_count = output.execute("SELECT COUNT(*) FROM monitoring_observations").fetchone()[0]
        non_grid_locations = output.execute(
            """SELECT COUNT(*) FROM monitoring_locations
               WHERE location_type NOT IN ('national_grid', 'user_point')"""
        ).fetchone()[0]
        if city_count != len(BASELINE_CITIES) or location_count != grid_count + city_count or non_grid_locations:
            raise ValueError("Public seed validation failed: expected only national grid and baseline city locations.")
        output.execute("VACUUM")

    return location_count, observation_count, grid_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/aqua_kz.db"))
    parser.add_argument("--output", type=Path, default=Path("data/public_seed.db"))
    args = parser.parse_args()
    locations, observations, cells = create_public_seed(args.source, args.output)
    print(
        f"Created {args.output}: {locations} national locations, "
        f"{cells} grid cells, {observations} observations."
    )


if __name__ == "__main__":
    main()
