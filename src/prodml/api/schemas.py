from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for a single prediction."""

    PU_DO: str = Field(
        ...,
        min_length=1,
        description="Pickup/drop-off location pair.",
        examples=["74_42"],
    )

    trip_distance: float = Field(
        ...,
        gt=0,
        description="Trip distance in miles.",
        examples=[2.5],
    )


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    duration: float = Field(
        ...,
        description="Predicted trip duration in minutes.",
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    trips: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: list[float]


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
