"""Drone inspection target route."""

from fastapi import APIRouter

from app.schemas import DroneTarget
from app.services.drone_service import build_drone_targets

router = APIRouter(prefix="/api", tags=["drone"])


@router.get("/drone-targets", response_model=list[DroneTarget])
def get_drone_targets() -> list[DroneTarget]:
    from app.services.satellite_service import load_sample_observations_with_risk

    return build_drone_targets(load_sample_observations_with_risk())
