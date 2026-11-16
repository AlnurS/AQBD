"""Local synthetic-data ML training and prediction routes."""

from fastapi import APIRouter, HTTPException, status

from app.schemas import ObservationInput, PredictionResponse, TrainingMetrics

router = APIRouter(prefix="/api", tags=["model"])


@router.post("/train-model", response_model=TrainingMetrics)
def train_model() -> TrainingMetrics:
    from app.ml.train_model import train_and_save_model

    return train_and_save_model()


@router.get("/model-metrics", response_model=TrainingMetrics)
def get_model_metrics() -> TrainingMetrics:
    from app.ml.train_model import load_saved_metrics

    return load_saved_metrics()


@router.post("/predict", response_model=PredictionResponse)
def predict(observation: ObservationInput) -> PredictionResponse:
    from app.ml.predict import ModelNotTrainedError, predict_drought

    try:
        return predict_drought(observation)
    except ModelNotTrainedError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
