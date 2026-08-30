# Module 4 — Serialization + ONNX

## Objective

Evaluate model serialization formats and compare the existing
scikit-learn/Pickle model with an ONNX Runtime representation.

The baseline model is a `LinearRegression` model trained using:

- `PU_DO`
- `trip_distance`

The fitted `DictVectorizer` is stored together with the model
inside the Pickle artifact.

---

## Artifacts

### Pickle

Existing artifact:

`models/baseline.pkl`

The Pickle artifact contains:

- fitted `LinearRegression`
- fitted `DictVectorizer`

### ONNX

Exported artifact:

`models/model.onnx`

The ONNX model accepts a dynamic batch dimension and 4,910
vectorized input features.

Input shape:

```text
[None, 4910]
