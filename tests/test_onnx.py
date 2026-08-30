import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import prepare_data


def test_onnx_prediction_parity() -> None:
    """Verify ONNX predictions match the pickle model."""

    # -------------------------
    # Load and prepare data
    # -------------------------
    data = load_data(settings.data_path)
    data = prepare_data(data)

    _, X_val, _, _ = split_data(data)

    # Use exactly 500 validation rows
    X_val = X_val.iloc[:500]

    # -------------------------
    # Load pickle artifact
    # -------------------------
    with open(settings.model_path, "rb") as file:
        artifact = pickle.load(file)

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    # Vectorize using the SAME fitted vectorizer
    val_dicts = X_val.to_dict(orient="records")

    X_val_vectorized = vectorizer.transform(val_dicts)

    # -------------------------
    # Pickle predictions
    # -------------------------
    pred_pkl = model.predict(X_val_vectorized)

    # -------------------------
    # Load ONNX model
    # -------------------------
    session = ort.InferenceSession(
        str(settings.onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # ONNX expects dense float32 input
    X_val_onnx = X_val_vectorized.toarray().astype(np.float32)

    # -------------------------
    # ONNX predictions
    # -------------------------
    pred_onnx = session.run(
        None,
        {
            input_name: X_val_onnx,
        },
    )[
        0
    ].reshape(-1)

    # -------------------------
    # Parity check
    # -------------------------
    assert pred_pkl.shape == pred_onnx.shape

    assert np.allclose(
        pred_pkl,
        pred_onnx,
        atol=1e-4,
    )
