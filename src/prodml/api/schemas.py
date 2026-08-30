from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str


class MetadataResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    training_date: str
    features: list[str]
    framework: str
    artifact_hash: str


class PredictionRequest(BaseModel):
    """Request schema for a single prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PU_DO": "74_42",
                "trip_distance": 2.5,
            }
        }
    )

    PU_DO: str = Field(
        min_length=1,
        description="Pickup and dropoff location pair.",
    )

    trip_distance: float = Field(
        gt=0,
        lt=200,
        description="Trip distance in miles.",
    )


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    trips: list[PredictionRequest] = Field(
        min_length=1,
        description="List of trips to predict.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trips": [
                    {
                        "PU_DO": "74_42",
                        "trip_distance": 2.5,
                    },
                    {
                        "PU_DO": "41_42",
                        "trip_distance": 5.0,
                    },
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float
