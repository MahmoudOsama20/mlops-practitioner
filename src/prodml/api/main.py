import hashlib
import logging
import time
from contextlib import asynccontextmanager
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
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)

configure_logging()

logger = logging.getLogger(__name__)

predictor = DurationPredictor()


def calculate_artifact_hash() -> str:
    """Calculate the SHA-256 hash of the model artifact."""

    sha256 = hashlib.sha256()

    with open(settings.model_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction model once when the API starts."""

    app.state.model_loaded = False

    try:
        predictor.load()

        app.state.model_loaded = True

        logger.info(
            "model_loaded",
            extra={
                "model_path": str(settings.model_path),
            },
        )

        yield

    except Exception:
        logger.exception("model_load_failure")
        raise

    finally:
        app.state.model_loaded = False


app = FastAPI(
    title="NYC Green Taxi Duration Prediction API",
    version=settings.model_version,
    description="API for predicting NYC Green Taxi trip duration.",
    lifespan=lifespan,
)


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


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request) -> HealthResponse:
    """Return healthy only when the model is loaded."""

    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return HealthResponse(status="ok")


@app.get(
    "/metadata",
    response_model=MetadataResponse,
)
def metadata() -> MetadataResponse:
    """Return model metadata."""

    artifact_hash = calculate_artifact_hash()

    return MetadataResponse(
        model_version=settings.model_version,
        training_date=settings.training_date,
        features=[
            "PU_DO",
            "trip_distance",
        ],
        framework=settings.framework,
        artifact_hash=artifact_hash,
    )


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

    start_time = time.perf_counter()

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

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "prediction_served",
        extra={
            "trip_distance": request.trip_distance,
            "duration_minutes": duration,
            "latency_ms": round(latency_ms, 3),
        },
    )

    return PredictionResponse(
        prediction=duration,
        model_version=settings.model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/predict/batch",
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

    logger.debug(
        "batch_feature_vector",
        extra={
            "batch_size": len(request.trips),
        },
    )

    start_time = time.perf_counter()

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

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "batch_prediction_served",
        extra={
            "batch_size": len(predictions),
            "latency_ms": round(latency_ms, 3),
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=settings.model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Return clean HTTP errors."""

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
    """Return clean Pydantic validation errors."""

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


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log unexpected errors without leaking tracebacks."""

    logger.exception(
        "unexpected_error",
        extra={
            "method": request.method,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
        },
        headers={
            "X-Request-ID": correlation_id_var.get(),
        },
    )
