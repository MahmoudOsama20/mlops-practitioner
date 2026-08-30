# Module 5 — Production API

## Objective

Expose the NYC Green Taxi trip-duration model through a
production-oriented FastAPI service.

The API provides health checking, model metadata, single
prediction, and batch prediction endpoints.

---

## API Architecture

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
