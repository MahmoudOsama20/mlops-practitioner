# Module 7 — Containerization and Publishing

## Objective

Containerize the NYC Green Taxi Duration Prediction API using Docker,
verify that the container runs correctly as a non-root user, compare
single-stage and multi-stage Docker builds, configure Docker Compose,
and publish the final image to Docker Hub.

---

## Project Containerization

The API is packaged using a multi-stage Docker build.

The Docker setup consists of:

- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `.dockerignore`

The final runtime image contains the installed application,
runtime dependencies, and model artifacts required by the API.

The build uses Python 3.11 slim as the base image.

---

## Multi-stage Docker Build

The Dockerfile uses two stages.

### Builder stage

The builder stage:

1. Uses `python:3.11-slim`.
2. Copies `pyproject.toml`.
3. Copies the `src/` package.
4. Installs the project and its dependencies into `/install`.

### Runtime stage

The runtime stage:

1. Uses a clean `python:3.11-slim` image.
2. Copies the installed application from the builder stage.
3. Copies the model artifacts into `/app/models`.
4. Configures the model paths through environment variables.
5. Creates and uses a non-root `appuser`.
6. Exposes port `8000`.
7. Defines a Docker health check.
8. Starts the FastAPI application with Uvicorn.

---

## Docker Image Size Comparison

Two Docker builds were measured.

### Multi-stage image

```text
Image: prodml-api:0.1.0
Disk usage: 1.11 GB
Content size: 251 MB
```

### Single-stage image

```text
Image: prodml-api:single
Disk usage: 1.13 GB
Content size: 256 MB
```

### Comparison

| Build type | Disk usage | Content size |
|---|---:|---:|
| Single-stage | 1.13 GB | 256 MB |
| Multi-stage | 1.11 GB | 251 MB |

The multi-stage image is smaller in the measured Docker environment.
The measured difference was approximately:

- 20 MB in disk usage
- 5 MB in content size

The multi-stage approach separates build-time installation from the
runtime image, preventing builder-stage contents from being copied
into the final runtime image.

---

## Docker Ignore

A root-level `.dockerignore` was added to prevent unnecessary files
from being sent to the Docker build context.

Excluded content includes:

- Git metadata
- Virtual environments
- Python cache files
- pytest cache and coverage files
- notebooks
- tests
- datasets
- local environment files
- IDE configuration

The `models/` directory is intentionally not excluded because the
runtime container needs the trained model artifacts.

---

## Model Artifacts

The Docker image contains the trained model artifacts:

```text
/app/models/baseline.pkl
/app/models/model.onnx
```

The application is configured to use:

```text
PRODML_MODEL_PATH=/app/models/baseline.pkl
PRODML_ONNX_PATH=/app/models/model.onnx
```

Docker Compose additionally mounts the model directory as read-only:

```text
../models:/app/models:ro
```

This prevents the application container from modifying the model
files through the mounted volume.

---

## Container Security

The API does not run as root.

A dedicated user was created:

```text
appuser
```

The container was verified with:

```powershell
docker compose -f docker/docker-compose.yml exec api whoami
```

which returned:

```text
appuser
```

Therefore, the application process runs under the non-root
`appuser` account.

---

## Docker Compose

Docker Compose is configured in:

```text
docker/docker-compose.yml
```

The Compose service provides:

- API service
- port mapping `8000:8000`
- model path environment variables
- model version and training date configuration
- read-only model volume
- restart policy

The service uses:

```yaml
restart: unless-stopped
```

---

## Container Verification

The container successfully started the FastAPI application and
loaded the model.

Startup logging confirmed:

```text
model_loaded
```

with the model path:

```text
/app/models/baseline.pkl
```

---

## Health Endpoint

The container was tested using:

```powershell
curl -i http://127.0.0.1:8000/health
```

The API returned:

```text
HTTP/1.1 200 OK
```

with:

```json
{
  "status": "ok"
}
```

---

## Prediction Endpoint

The prediction endpoint was tested using:

```json
{
  "PU_DO": "74_42",
  "trip_distance": 2.5
}
```

The container returned:

```text
HTTP/1.1 200 OK
```

with the prediction:

```text
12.727868515169943
```

The response also included:

- model version `0.1.0`
- correlation ID
- request ID through the `X-Request-ID` header
- prediction latency

Example response:

```json
{
  "prediction": 12.727868515169943,
  "model_version": "0.1.0",
  "correlation_id": "...",
  "latency_ms": 4.834
}
```

---

## Published Docker Image

The image was tagged using semantic versioning:

```text
mahmoudosama20/prodml-api:0.1.0
mahmoudosama20/prodml-api:latest
```

Docker Hub authentication was successful using the configured
Docker credentials.

The versioned image was used to run the API successfully.

---

## Reproducibility

The final Docker image packages the API and its runtime dependencies,
so the application can be started without recreating the local Python
virtual environment.

The intended runtime command is:

```powershell
docker run --rm -p 8000:8000 mahmoudosama20/prodml-api:0.1.0
```

Once running, the API is available on:

```text
http://127.0.0.1:8000
```

The health endpoint can be checked with:

```powershell
curl http://127.0.0.1:8000/health
```

A prediction can be requested with:

```powershell
curl -i -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"PU_DO":"74_42","trip_distance":2.5}'
```

---

## Conclusion

The NYC Green Taxi Duration Prediction API has been containerized
using a multi-stage Docker build.

The container:

- builds successfully
- loads the trained model
- runs as a non-root user
- exposes a health endpoint
- serves predictions successfully
- preserves structured logging and request correlation
- supports Docker Compose
- uses read-only model mounting in Compose
- is packaged as a versioned Docker image
- is published under the `mahmoudosama20/prodml-api` repository

The measured multi-stage image was smaller than the single-stage
image in the Docker environment used for this project.
