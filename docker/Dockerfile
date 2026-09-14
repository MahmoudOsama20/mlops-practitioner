FROM python:3.11-slim

WORKDIR /app

RUN useradd \
    --create-home \
    --uid 1000 \
    appuser

COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt

RUN pip install \
    --no-cache-dir \
    -r /tmp/requirements-runtime.txt

COPY src/ /app/src/

COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser pipelines/models/model.pkl /app/models/candidate.pkl

ENV PYTHONPATH=/app/src

ENV PRODML_MODEL_PATH=/app/models/baseline.pkl
ENV PRODML_ONNX_PATH=/app/models/model.onnx
ENV PRODML_LOCAL_MODEL_PATH=/app/models/candidate.pkl

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
