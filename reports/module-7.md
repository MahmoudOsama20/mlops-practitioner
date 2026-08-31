# Module 7 — Containerization and Publishing

## Objective

Containerize the NYC Green Taxi Duration Prediction API using Docker,
verify that the container runs correctly as a non-root user, compare
single-stage and multi-stage Docker builds, configure Docker Compose,
and publish the final image to Docker Hub.

---

## 1. Project Containerization

The API is packaged using a multi-stage Docker build.

The Docker setup consists of:

```text
docker/Dockerfile
docker/docker-compose.yml
.dockerignore
```

The final runtime image contains:

- the installed application
- runtime dependencies
- trained model artifacts required by the API

The Docker build uses:

```text
python:3.11-slim
```

as its base image.

---

## 2. Multi-stage Docker Build

The Dockerfile uses two stages:

```text
Builder Stage
     |
     v
Runtime Stage
     |
     v
FastAPI Application
```

### Builder Stage

The builder stage:

1. Uses `python:3.11-slim`.
2. Copies `pyproject.toml`.
3. Copies the `src/` package.
4. Installs the project and dependencies into `/install`.

The builder stage is responsible for creating the installed application
environment without carrying its build contents into the final runtime
image.

### Runtime Stage

The runtime stage:

1. Uses a clean `python:3.11-slim` image.
2. Copies the installed application from the builder stage.
3. Copies model artifacts into `/app/models`.
4. Configures model paths through environment variables.
5. Creates and uses a non-root `appuser`.
6. Exposes port `8000`.
7. Defines a Docker health check.
8. Starts the FastAPI application with Uvicorn.

---

## 3. Docker Image Size Comparison

Two Docker builds were measured.

### Multi-stage Image

```text
Image: prodml-api:0.1.0

Disk usage: 1.11 GB
Content size: 251 MB
```

### Single-stage Image

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

The multi-stage image was smaller in the measured Docker environment.

Measured difference:

```text
Disk usage:  approximately 20 MB
Content size: approximately 5 MB
```

The multi-stage approach separates build-time installation from the
runtime image, preventing builder-stage contents from being copied into
the final runtime image.

---

## 4. Docker Ignore

A root-level `.dockerignore` was added to prevent unnecessary files from
being sent to the Docker build context.

Excluded content includes:

- Git metadata
- virtual environments
- Python cache files
- pytest cache
- coverage files
- notebooks
- tests
- datasets
- local environment files
- IDE configuration

The `models/` directory is intentionally not excluded because the
runtime container needs the trained model artifacts.

This keeps the build context focused on files required to construct and
run the service.

---

## 5. Model Artifacts

The Docker image contains:

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

The read-only mount prevents the application container from modifying
the model files through the mounted volume.

---

## 6. Container Security

The API does not run as root.

A dedicated user was created:

```text
appuser
```

The running Compose service was verified with:

```powershell
docker compose -f docker/docker-compose.yml exec api whoami
```

The command returned:

```text
appuser
```

Therefore, the application process runs under the non-root `appuser`
account.

This provides a basic container-security improvement over running the
application as root.

---

## 7. Docker Compose

Docker Compose is configured in:

```text
docker/docker-compose.yml
```

The Compose service provides:

- API service
- port mapping `8000:8000`
- model path environment variables
- model version configuration
- training date configuration
- read-only model volume
- restart policy

The service uses:

```yaml
restart: unless-stopped
```

The Compose configuration was used to verify both model availability and
non-root execution.

---

## 8. Container Verification

The container successfully started the FastAPI application and loaded
the trained model.

Startup logging confirmed:

```text
model_loaded
```

with the model path:

```text
/app/models/baseline.pkl
```

The container then successfully served API requests.

---

## 9. Health Endpoint

The container was tested using:

```powershell
curl.exe -i http://127.0.0.1:8000/health
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

The response also included an `X-Request-ID` header, demonstrating that
the API's correlation middleware remains active inside the container.

---

## 10. Prediction Endpoint

The prediction endpoint was tested with:

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
- `X-Request-ID` response header
- prediction latency

Example:

```json
{
  "prediction": 12.727868515169943,
  "model_version": "0.1.0",
  "correlation_id": "...",
  "latency_ms": 4.834
}
```

The container logs also confirmed the request lifecycle:

```text
request_started
function_timing
prediction_served
request_completed
```

---

## 11. Published Docker Image

The image was published under:

```text
mahmoudosama20/prodml-api
```

Available tags:

```text
mahmoudosama20/prodml-api:0.1.0
mahmoudosama20/prodml-api:latest
```

The versioned image was successfully pushed and subsequently pulled
from Docker Hub.

### Pull Verification

```powershell
docker pull mahmoudosama20/prodml-api:0.1.0
```

The registry returned the image digest:

```text
sha256:29112bd77ba93ee0687361088b316a21b4dc303af1be92af5296b640a3b9e920
```

The `latest` tag was also successfully pulled.

Docker Hub repository:

```text
https://hub.docker.com/r/mahmoudosama20/prodml-api
```

---

## 12. Reproducibility

The final Docker image packages the API and its runtime dependencies,
allowing the application to run without recreating the local Python
virtual environment.

The intended runtime command is:

```powershell
docker run --rm -p 8000:8000 mahmoudosama20/prodml-api:0.1.0
```

Once running, the API is available on:

```text
http://127.0.0.1:8000
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Prediction:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"PU_DO":"74_42","trip_distance":2.5}'
```

The same container image can therefore be pulled from Docker Hub and run
without installing the project into a local Python environment.

---

## 13. Docker Build

The final multi-stage image was built with:

```powershell
docker build -f docker/Dockerfile -t prodml-api:0.1.0 .
```

The build completed successfully.

The image was then tagged as:

```powershell
docker tag prodml-api:0.1.0 prodml-api:latest
```

Both tags point to the same published image release.

---

## 14. Docker Image Contents

The runtime container contains the application and model artifacts
required for inference.

Verified model directory:

```text
/app/models
├── baseline.pkl
└── model.onnx
```

Observed sizes:

```text
baseline.pkl   approximately 149 KB
model.onnx     approximately 25 KB
```

The container successfully loaded `baseline.pkl` during application
startup.

---

## 15. Container Runtime

The container starts Uvicorn using the FastAPI application.

Runtime configuration:

```text
Host: 0.0.0.0
Port: 8000
User: appuser
```

Successful startup produced:

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

The service then accepted requests from the host through the published
port.

---

## 16. Production-Oriented Container Features

The final containerization implementation includes:

- multi-stage Docker build
- Python 3.11 slim runtime
- isolated runtime image
- trained model artifacts
- ONNX artifact
- environment-based model configuration
- non-root execution
- Docker health check
- port `8000`
- Docker Compose support
- read-only model volume
- restart policy
- versioned image tags
- Docker Hub publication

---

## 17. Security Considerations

The containerization work includes several basic security measures:

### Non-root execution

The application runs as:

```text
appuser
```

instead of root.

### Read-only model volume

Docker Compose mounts:

```text
../models:/app/models:ro
```

which prevents the service from writing to the mounted model directory.

### Reduced build context

`.dockerignore` prevents unnecessary local files such as:

```text
.git
.venv
__pycache__
.pytest_cache
coverage files
tests
notebooks
datasets
.env
```

from being included in the build context.

These measures reduce unnecessary container privileges and runtime
write access.

---

## 18. Single-stage vs Multi-stage

The two approaches were evaluated empirically.

### Single-stage

```text
prodml-api:single

Disk usage: 1.13 GB
Content size: 256 MB
```

### Multi-stage

```text
prodml-api:0.1.0

Disk usage: 1.11 GB
Content size: 251 MB
```

### Result

The multi-stage image is smaller in the measured environment:

```text
Approximately 20 MB less disk usage
Approximately 5 MB less content size
```

More importantly, the multi-stage design keeps build-stage contents out
of the runtime image.

For this project, the multi-stage build was therefore selected as the
final containerization strategy.

---

## 19. Container Verification Checklist

The final container was verified through:

```text
Build
  ↓
Run
  ↓
Model loading
  ↓
Non-root user verification
  ↓
Health request
  ↓
Prediction request
  ↓
Structured logs
  ↓
Docker Hub pull
```

Observed results:

```text
Docker build              → SUCCESS
Container startup         → SUCCESS
Model loading             → SUCCESS
whoami                    → appuser
GET /health               → 200 OK
POST /predict             → 200 OK
Docker Hub push           → SUCCESS
Docker Hub pull           → SUCCESS
```

---

## 20. Integration with Previous Modules

Module 7 packages the work completed in the earlier modules into a
portable container.

The resulting flow is:

```text
Baseline Model
      |
      v
Model Serialization
      |
      v
Production FastAPI API
      |
      v
Docker Container
      |
      v
Docker Hub
```

The container therefore packages the existing model-serving system
rather than introducing a separate inference implementation.

---

## 21. Conclusion

The NYC Green Taxi Duration Prediction API has been successfully
containerized using a multi-stage Docker build.

The final container:

- builds successfully
- contains the trained model artifacts
- loads the model successfully
- runs as a non-root user
- exposes port `8000`
- provides a Docker health check
- serves the `/health` endpoint successfully
- serves the `/predict` endpoint successfully
- preserves structured JSON logging
- preserves request correlation
- supports Docker Compose
- uses a read-only model volume in Compose
- uses environment-based model configuration
- is available through versioned Docker tags
- is published to Docker Hub

The measured multi-stage image was smaller than the single-stage image
in the Docker environment used for this project.

The final published image is:

```text
mahmoudosama20/prodml-api:0.1.0
```

---

## 22. Definition of Done

- [x] Dockerfile implemented
- [x] Multi-stage Docker build implemented
- [x] Builder stage implemented
- [x] Runtime stage implemented
- [x] Python 3.11 slim used
- [x] Application installed into runtime image
- [x] Model artifacts copied into the image
- [x] Model environment variables configured
- [x] Non-root `appuser` created
- [x] Non-root execution verified
- [x] Port `8000` exposed
- [x] Docker health check configured
- [x] Uvicorn startup configured
- [x] `.dockerignore` implemented
- [x] Single-stage image built
- [x] Multi-stage image built
- [x] Image size comparison completed
- [x] Docker Compose configured
- [x] Read-only model volume configured
- [x] Compose restart policy configured
- [x] Container startup verified
- [x] Model loading verified
- [x] `/health` verified
- [x] `/predict` verified
- [x] Structured logging verified in container
- [x] Docker Hub authentication completed
- [x] Versioned image published
- [x] `latest` image tag published
- [x] Published image pulled successfully
- [x] Reproducible Docker run command documented

---

## 23. Module Status

**Module 7 — Complete**

The project now has a reproducible, containerized API that can be
distributed through Docker Hub and executed independently of the local
Python development environment.

Final published image:

```text
mahmoudosama20/prodml-api:0.1.0
```

Latest tag:

```text
mahmoudosama20/prodml-api:latest
```
