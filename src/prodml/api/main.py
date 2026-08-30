import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.config import settings
from prodml.logging_conf import configure_logging, correlation_id_var
from prodml.predict import DurationPredictor

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NYC Green Taxi Duration Prediction API",
    version="0.1.0",
    description="API for predicting NYC Green Taxi trip duration.",
)


predictor = DurationPredictor()


@app.middleware("http")
async def correlation_middleware(
    request: Request,
    call_next,
):
    """Attach a correlation ID to every request."""

    correlation_id = str(uuid4())

    token = correlation_id_var.set(correlation_id)

    logger.info(
        "request_started",
        extra={
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = correlation_id

        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            },
        )

        return response

    finally:
        correlation_id_var.reset(token)


@app.on_event("startup")
def load_prediction_model() -> None:
    """Load the prediction model when the API starts."""

    try:
        predictor.load()

        logger.info(
            "model_loaded",
            extra={
                "model_path": str(settings.model_path),
            },
        )

    except Exception:
        logger.exception("model_load_failure")
        raise


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return the health status of the API."""

    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """Predict the duration of a single taxi trip."""

    if request.trip_distance > 100:
        logger.warning(
            "input_outside_training_range",
            extra={
                "trip_distance": request.trip_distance,
            },
        )

        raise HTTPException(
            status_code=422,
            detail="trip_distance must be <= 100 miles",
        )

    logger.debug(
        "feature_vector",
        extra={
            "PU_DO": request.PU_DO,
            "trip_distance": request.trip_distance,
        },
    )

    try:
        duration = predictor.predict_one(
            {
                "PU_DO": request.PU_DO,
                "trip_distance": request.trip_distance,
            }
        )

    except Exception:
        logger.exception("prediction_failure")
        raise

    logger.info(
        "prediction_served",
        extra={
            "trip_distance": request.trip_distance,
            "duration_minutes": duration,
        },
    )

    return PredictionResponse(
        duration=duration,
    )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict the duration of multiple taxi trips."""

    for trip in request.trips:
        if trip.trip_distance > 100:
            logger.warning(
                "input_outside_training_range",
                extra={
                    "trip_distance": trip.trip_distance,
                },
            )

            raise HTTPException(
                status_code=422,
                detail="trip_distance must be <= 100 miles",
            )

    try:
        predictions = predictor.predict_batch(
            [
                {
                    "PU_DO": trip.PU_DO,
                    "trip_distance": trip.trip_distance,
                }
                for trip in request.trips
            ]
        )

    except Exception:
        logger.exception("batch_prediction_failure")
        raise

    logger.info(
        "batch_prediction_served",
        extra={
            "batch_size": len(predictions),
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Log rejected API requests."""

    logger.error(
        "validation_rejection",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
        },
        headers={
            "X-Request-ID": correlation_id_var.get(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Log FastAPI validation failures."""

    logger.error(
        "validation_rejection",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": 422,
            "errors": exc.errors(),
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
        headers={
            "X-Request-ID": correlation_id_var.get(),
        },
    )
