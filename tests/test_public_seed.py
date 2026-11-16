import sqlite3
import shutil
from pathlib import Path

from app.services import inference_run_service, monitoring_service
from scripts.create_public_seed_db import BASELINE_CITIES, add_baseline_city_locations


def test_public_seed_exposes_baseline_cities_as_monitoring_locations(monkeypatch, tmp_path) -> None:
    database_path = tmp_path / "aqua_kz.db"
    seed_path = Path(__file__).parents[1] / "data" / "public_seed.db"
    shutil.copy2(seed_path, database_path)
    monkeypatch.setattr(inference_run_service, "DB_PATH", database_path)
    monkeypatch.setattr(inference_run_service, "PREDICTIONS_DIR", tmp_path / "predictions")

    monitoring_service.initialize_monitoring_storage()
    locations = monitoring_service.list_locations()

    assert len(locations) == len(BASELINE_CITIES)
    assert {location["name"] for location in locations} == {city[0] for city in BASELINE_CITIES}
    assert next(location for location in locations if location["name"] == "Рудный")["latitude"] == 52.9529


def test_baseline_city_points_include_rudny_and_are_idempotent() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            """CREATE TABLE monitoring_locations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                region TEXT,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL,
                location_type TEXT NOT NULL CHECK (location_type IN ('user_point', 'national_grid'))
            )"""
        )
        connection.execute(
            "CREATE UNIQUE INDEX active_user_point_coordinates "
            "ON monitoring_locations(latitude, longitude) WHERE active = 1 AND location_type = 'user_point'"
        )

        assert add_baseline_city_locations(connection) == len(BASELINE_CITIES)
        assert add_baseline_city_locations(connection) == 0

        cities = connection.execute(
            """SELECT name, latitude, longitude, region, location_type
               FROM monitoring_locations"""
        ).fetchall()

    assert len(cities) == len(BASELINE_CITIES)
    rudny = next(city for city in cities if city[0] == "Рудный")
    assert rudny[1:3] == (52.9529, 63.1308)
    assert rudny[3:] == ("Базовая точка по Казахстану", "user_point")
