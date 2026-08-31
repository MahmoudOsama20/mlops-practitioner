# Module 5 — Production API

## Objective

Expose the NYC Green Taxi trip-duration model through a
production-oriented FastAPI service.

The API provides:

- health checking
- model metadata
- single prediction
- batch prediction
- request validation
- structured JSON logging
- correlation IDs
- request latency measurement
- error handling

---

## 1. API Architecture

The implemented request flow is:

```text
HTTP Request
     |
     v
Correlation Middleware
     |
     +--> X-Request-ID
     |
     v
FastAPI Endpoint
     |
     v
Pydantic Validation
     |
     v
DurationPredictor
     |
     v
Prediction
     |
     v
Structured JSON Response
```

Each request receives a unique correlation ID. The ID is propagated
through the request logs and returned to the client through the
`X-Request-ID` response header.

---

## 2. Application Structure

The API is implemented under:

```text
src/prodml/api/
├── __init__.py
├── main.py
└── schemas.py
```

### `main.py`

Responsible for:

- creating the FastAPI application
- configuring logging
- loading the prediction model
- implementing middleware
- exposing API endpoints
- handling validation errors
- handling HTTP exceptions
- logging request lifecycle events

### `schemas.py`

Contains the Pydantic request and response models used by the API.

---

## 3. Model Loading

The prediction model is loaded during application startup.

The API uses:

```text
DurationPredictor
```

The startup process loads the trained model artifact before serving
requests.

Successful startup produces a structured log event similar to:

```json
{
  "level": "INFO",
  "logger": "prodml.api.main",
  "message": "model_loaded",
  "model_path": "/app/models/baseline.pkl"
}
```

Loading the model at startup avoids loading the model separately for
each prediction request.

---

## 4. Correlation Middleware

The API creates a UUID correlation ID for every incoming request.

The middleware:

1. Generates a correlation ID.
2. Stores it in the logging context.
3. Logs `request_started`.
4. Executes the endpoint.
5. Adds `X-Request-ID` to the response.
6. Logs `request_completed`.
7. Resets the logging context.

Example:

```text
request_started
    correlation_id = 346be0a2-0dba-4bb4-9d81-01cb9d488b63

prediction_served
    correlation_id = 346be0a2-0dba-4bb4-9d81-01cb9d488b63

request_completed
    correlation_id = 346be0a2-0dba-4bb4-9d81-01cb9d488b63
```

This makes it possible to connect logs belonging to the same request.

---

## 5. Structured Logging

The service uses structured JSON logging.

Important events include:

```text
model_loaded
model_load_failure
request_started
request_completed
prediction_served
batch_prediction_served
validation_rejection
prediction_failure
batch_prediction_failure
function_timing
input_outside_training_range
```

Example successful prediction log:

```json
{
  "timestamp": "2026-08-31T07:04:24Z",
  "level": "INFO",
  "logger": "prodml.api.main",
  "message": "prediction_served",
  "correlation_id": "346be0a2-0dba-4bb4-9d81-01cb9d488b63",
  "trip_distance": 2.5,
  "duration_minutes": 12.727868515169943,
  "latency_ms": 3.445
}
```

---

## 6. API Endpoints

The service exposes four main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/metadata` | Model metadata |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction |

---

## 7. Health Endpoint

### Request

```text
GET /health
```

Example:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Observed response:

```text
HTTP/1.1 200 OK
```

```json
{
  "status": "ok"
}
```

The response also contains:

```text
X-Request-ID
```

which confirms that the correlation middleware is applied to the
request.

---

## 8. Metadata Endpoint

### Request

```text
GET /metadata
```

Example:

```powershell
curl.exe -i http://127.0.0.1:8000/metadata
```

Observed response:

```json
{
  "model_version": "0.1.0",
  "training_date": "2026-08-30",
  "features": [
    "PU_DO",
    "trip_distance"
  ],
  "framework": "scikit-learn",
  "artifact_hash": "c362d9d8622bf5a528dc17c92783518a3c81a216fc048c5b8760ddd96db9218a"
}
```

The metadata provides basic model identity and artifact traceability.

---

## 9. Single Prediction Endpoint

### Request

```text
POST /predict
```

Example request:

```json
{
  "PU_DO": "74_42",
  "trip_distance": 2.5
}
```

Example command:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"PU_DO":"74_42","trip_distance":2.5}'
```

Observed response:

```text
HTTP/1.1 200 OK
```

```json
{
  "prediction": 12.727868515169943,
  "model_version": "0.1.0",
  "correlation_id": "f3643606-0f14-4380-8578-bb4cf2054acd",
  "latency_ms": 4.834
}
```

The response includes:

- prediction
- model version
- correlation ID
- inference latency

---

## 10. Batch Prediction Endpoint

### Request

```text
POST /predict/batch
```

Example request:

```json
{
  "trips": [
    {
      "PU_DO": "74_42",
      "trip_distance": 2.5
    },
    {
      "PU_DO": "41_42",
      "trip_distance": 5.0
    }
  ]
}
```

Example command:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict/batch `
  -H "Content-Type: application/json" `
  -d '{"trips":[{"PU_DO":"74_42","trip_distance":2.5},{"PU_DO":"41_42","trip_distance":5.0}]}'
```

Observed response:

```json
{
  "predictions": [
    12.727868515169943,
    19.903236786100038
  ],
  "model_version": "0.1.0",
  "correlation_id": "c04d0843-f9f0-492c-bdb2-2d3c72e61d57",
  "latency_ms": 0.572
}
```

The batch endpoint returns one prediction for each input trip.

---

## 11. Request Validation

Pydantic models validate API input before it reaches the prediction
logic.

The prediction request includes:

```text
PU_DO: non-empty string
trip_distance: greater than 0
```

The API also enforces the supported training range:

```text
trip_distance <= 100 miles
```

### Negative Distance

Request:

```json
{
  "PU_DO": "74_42",
  "trip_distance": -5
}
```

Observed result:

```text
HTTP 422 Unprocessable Entity
```

with a Pydantic validation error indicating that the value must be
greater than `0`.

### Outside Training Range

Request:

```json
{
  "PU_DO": "74_42",
  "trip_distance": 101
}
```

Observed result:

```text
HTTP 422 Unprocessable Entity
```

with:

```json
{
  "detail": "trip_distance must be <= 100 miles"
}
```

These validations prevent the service from accepting values outside the
range used by the baseline training pipeline.

---

## 12. Error Handling

The API includes explicit exception handlers for:

- `HTTPException`
- `RequestValidationError`

Validation rejections are logged as structured events.

Unexpected prediction errors are logged with exception information and
propagated as server errors.

The API test suite includes an unexpected-error scenario to verify this
behavior.

---

## 13. Latency Measurement

Prediction latency is measured for API inference.

The response exposes:

```text
latency_ms
```

The predictor also emits a structured timing event.

Example:

```json
{
  "message": "function_timing",
  "function": "predict_one",
  "duration_ms": 3.32
}
```

This provides a basic observability signal for inference performance.

---

## 14. Response Tracing

Each API response contains an:

```text
X-Request-ID
```

header.

For prediction responses, the same identifier is also included in the
JSON body as:

```json
{
  "correlation_id": "..."
}
```

This allows a client to correlate an API response with the associated
structured server logs.

---

## 15. Automated API Tests

The API is covered by:

```text
tests/test_api.py
```

The test suite currently contains seven API tests:

```text
test_health
test_metadata
test_predict
test_predict_batch
test_predict_negative_distance
test_predict_distance_outside_training_range
test_predict_unexpected_error
```

Final API test result:

```text
7 passed
```

---

## 16. Full Test Suite

The API tests are part of the complete project test suite.

The final suite contains:

```text
17 tests
```

Result:

```text
17 passed
```

The complete suite covers:

- API behavior
- feature engineering
- prediction behavior
- deterministic predictions
- batch predictions
- ONNX serialization parity

---

## 17. Coverage

The complete project currently reports:

```text
304 statements
73 missed
75.99% coverage
```

The configured minimum is:

```text
70%
```

Therefore the coverage requirement is satisfied:

```text
75.99% > 70%
```

The API module itself has:

```text
91% coverage
```

according to the final coverage report.

---

## 18. API Validation in a Running Environment

The service was manually verified while running locally.

### Health

```text
GET /health
→ 200 OK
```

### Metadata

```text
GET /metadata
→ 200 OK
```

### Single prediction

```text
POST /predict
→ 200 OK
```

### Batch prediction

```text
POST /predict/batch
→ 200 OK
```

### Invalid negative distance

```text
POST /predict
→ 422 Unprocessable Entity
```

### Distance outside training range

```text
POST /predict
→ 422 Unprocessable Entity
```

The API was also successfully executed inside the Docker container.

---

## 19. Dockerized API

The FastAPI service is packaged into a Docker image.

Build:

```powershell
docker build -f docker/Dockerfile -t prodml-api:0.1.0 .
```

Run:

```powershell
docker run --rm -p 8000:8000 prodml-api:0.1.0
```

The container starts Uvicorn on:

```text
0.0.0.0:8000
```

and successfully loads:

```text
/app/models/baseline.pkl
```

The API was verified from the host through the published port.

---

## 20. Container Security

The Docker image creates and uses a non-root user:

```text
appuser
```

The running container was verified with:

```text
whoami
```

which returned:

```text
appuser
```

This avoids running the application as root inside the container.

---

## 21. Docker Compose

The project also provides:

```text
docker/docker-compose.yml
```

Compose was used to verify the API and model artifacts.

The running container contained:

```text
/app/models/baseline.pkl
/app/models/model.onnx
```

and executed as:

```text
appuser
```

---

## 22. Production-Oriented Features

The API goes beyond a minimal prediction endpoint by providing:

- model loading at startup
- typed request and response schemas
- input validation
- batch inference
- health checks
- model metadata
- structured JSON logs
- correlation IDs
- response request IDs
- inference latency measurement
- explicit validation error handling
- unexpected-error logging
- Docker packaging
- non-root container execution

These capabilities establish the API as a production-oriented service
rather than a notebook-only prediction interface.

---

## 23. Current API Architecture

```text
                         HTTP Client
                              |
                              v
                   ┌────────────────────┐
                   │ Correlation         │
                   │ Middleware          │
                   │                    │
                   │ Generate UUID       │
                   │ Set logging context │
                   │ Add X-Request-ID    │
                   └─────────┬──────────┘
                             |
                             v
                   ┌────────────────────┐
                   │     FastAPI        │
                   │    Endpoint       │
                   └─────────┬──────────┘
                             |
                             v
                   ┌────────────────────┐
                   │ Pydantic Schemas   │
                   │                    │
                   │ Type validation    │
                   │ Range validation   │
                   └─────────┬──────────┘
                             |
                             v
                   ┌────────────────────┐
                   │ DurationPredictor  │
                   │                    │
                   │ Load model         │
                   │ Vectorize features │
                   │ Predict             │
                   └─────────┬──────────┘
                             |
                             v
                   ┌────────────────────┐
                   │ Prediction         │
                   │ + model version    │
                   │ + correlation ID   │
                   │ + latency          │
                   └─────────┬──────────┘
                             |
                             v
                   ┌────────────────────┐
                   │ Structured JSON    │
                   │ HTTP Response      │
                   └────────────────────┘
```

---

## 24. Module 5 Conclusion

Module 5 successfully exposes the NYC Green Taxi duration model through
a production-oriented FastAPI service.

The API provides:

```text
/health
/metadata
/predict
/predict/batch
```

The implementation includes the required architecture:

```text
HTTP Request
     ↓
Correlation Middleware
     ↓
FastAPI Endpoint
     ↓
Pydantic Validation
     ↓
DurationPredictor
     ↓
Prediction
     ↓
Structured JSON Response
```

The service was validated through automated tests and manual requests.

Final API test result:

```text
7 passed
```

Final project test result:

```text
17 passed
```

Final project coverage:

```text
75.99%
```

The API also runs successfully inside the Dockerized deployment and
uses structured request tracing and non-root container execution.

---

## 25. Definition of Done

- [x] FastAPI service implemented
- [x] Correlation middleware implemented
- [x] `X-Request-ID` response header implemented
- [x] Pydantic request validation implemented
- [x] Single prediction endpoint implemented
- [x] Batch prediction endpoint implemented
- [x] Health endpoint implemented
- [x] Metadata endpoint implemented
- [x] Model loaded during application startup
- [x] Structured JSON logging implemented
- [x] Correlation IDs included in logs
- [x] Prediction latency measured
- [x] Input range validation implemented
- [x] Validation errors handled
- [x] Unexpected prediction errors logged
- [x] API tests implemented
- [x] Seven API tests passing
- [x] Full test suite passing
- [x] Coverage requirement satisfied
- [x] Dockerized API verified
- [x] Non-root container execution verified
- [x] Docker Compose verified

---

## 26. Module Status

**Module 5 — Complete**

The project now exposes the trained NYC Green Taxi duration model
through a validated, observable, containerized FastAPI service with
single and batch prediction capabilities, model metadata, health
checking, structured logging, request tracing, and automated tests.
