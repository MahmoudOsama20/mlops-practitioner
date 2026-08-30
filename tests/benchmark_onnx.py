import pickle
import time

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import prepare_data

N_ROWS = 500
N_RUNS = 100


def percentile_95(values: list[float]) -> float:
    """Return the 95th percentile latency."""
    return float(np.percentile(values, 95))


def main() -> None:
    """Benchmark pickle and ONNX inference."""

    # -------------------------
    # Prepare the same 500 rows
    # -------------------------
    data = load_data(settings.data_path)
    data = prepare_data(data)

    _, X_val, _, _ = split_data(data)

    X_val = X_val.iloc[:N_ROWS]

    # -------------------------
    # Load pickle artifact
    # -------------------------
    with open(settings.model_path, "rb") as file:
        artifact = pickle.load(file)

    pickle_model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    X_val_vectorized = vectorizer.transform(X_val.to_dict(orient="records"))

    # -------------------------
    # Load ONNX model
    # -------------------------
    session = ort.InferenceSession(
        str(settings.onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    X_val_onnx = X_val_vectorized.toarray().astype(np.float32)

    # -------------------------
    # Warm-up
    # -------------------------
    pickle_model.predict(X_val_vectorized)

    session.run(
        None,
        {input_name: X_val_onnx},
    )

    # -------------------------
    # Benchmark Pickle
    # -------------------------
    pickle_latencies = []

    for _ in range(N_RUNS):
        start = time.perf_counter()

        pickle_model.predict(X_val_vectorized)

        elapsed = (time.perf_counter() - start) * 1000

        pickle_latencies.append(elapsed)

    # -------------------------
    # Benchmark ONNX
    # -------------------------
    onnx_latencies = []

    for _ in range(N_RUNS):
        start = time.perf_counter()

        session.run(
            None,
            {input_name: X_val_onnx},
        )

        elapsed = (time.perf_counter() - start) * 1000

        onnx_latencies.append(elapsed)

    # -------------------------
    # Results
    # -------------------------
    pickle_mean = float(np.mean(pickle_latencies))

    pickle_p95 = percentile_95(pickle_latencies)

    onnx_mean = float(np.mean(onnx_latencies))

    onnx_p95 = percentile_95(onnx_latencies)

    print(f"Pickle mean latency: {pickle_mean:.3f} ms")

    print(f"Pickle p95 latency: {pickle_p95:.3f} ms")

    print(f"ONNX mean latency: {onnx_mean:.3f} ms")

    print(f"ONNX p95 latency: {onnx_p95:.3f} ms")


if __name__ == "__main__":
    main()
