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

## 1. Artifacts

### Pickle

Existing artifact:

```text
models/baseline.pkl
```

The Pickle artifact contains:

- fitted `LinearRegression`
- fitted `DictVectorizer`

This artifact is the reference implementation for the model.

> Pickle artifacts should only be loaded from trusted sources because
> Python Pickle deserialization can execute arbitrary code.

### ONNX

Exported artifact:

```text
models/model.onnx
```

The ONNX model provides an alternative representation for inference
using ONNX Runtime.

The model accepts a dynamic batch dimension and 4,910 vectorized
input features.

Input shape:

```text
[None, 4910]
```

The exported model was successfully inspected with ONNX Runtime.

---

## 2. Model Input and Output

The exported ONNX Runtime session reports:

```text
Input name:  input
Input shape: [None, 4910]
Output name: variable
```

The `None` dimension represents a dynamic batch size.

The 4,910 features correspond to the vectorized feature representation
produced from the baseline feature set.

The baseline feature set is:

```text
PU_DO
trip_distance
```

---

## 3. Export Process

The ONNX export is implemented in:

```text
src/prodml/export.py
```

The workflow is:

```text
baseline.pkl
     │
     ├── LinearRegression
     │
     └── DictVectorizer
             │
             ▼
      Vectorized features
             │
             ▼
        ONNX export
             │
             ▼
      models/model.onnx
```

The Pickle artifact remains available as the reference model while the
ONNX artifact is used to evaluate an alternative model representation.

---

## 4. ONNX Runtime Verification

The generated ONNX model was loaded with ONNX Runtime using the CPU
execution provider.

Verification confirmed:

```text
Input:
    input

Shape:
    [None, 4910]

Output:
    variable
```

This confirms that the generated artifact is readable by ONNX Runtime
and exposes the expected input/output interface.

---

## 5. Prediction Parity

A dedicated test verifies prediction parity between the reference
model and the ONNX representation.

Test file:

```text
tests/test_serialization.py
```

Test:

```text
test_onnx_prediction_parity
```

Result:

```text
PASSED
```

The complete test suite also includes this serialization test.

This provides a regression check that exporting the model to ONNX does
not materially change the prediction behavior used by the baseline.

---

## 6. Inference Benchmark

A local inference benchmark was executed to compare the reference
Pickle/scikit-learn implementation with ONNX Runtime.

### Results

| Runtime | Mean Latency | P95 Latency |
|---|---:|---:|
| Pickle / scikit-learn | 0.084 ms | 0.101 ms |
| ONNX Runtime | 0.223 ms | 0.231 ms |

### Observations

For this baseline model and local benchmark:

- Pickle/scikit-learn had the lower mean latency.
- Pickle/scikit-learn had the lower P95 latency.
- ONNX Runtime was slower for this particular workload.
- The benchmark therefore does not show a latency advantage for ONNX
  on this small model.

The benchmark results should be interpreted as local measurements for
this implementation and workload rather than as a general statement
that one runtime is always faster than the other.

---

## 7. Pickle vs ONNX

| Aspect | Pickle | ONNX |
|---|---|---|
| Current artifact | `models/baseline.pkl` | `models/model.onnx` |
| Reference model | Yes | No |
| Portable model representation | Limited to Python ecosystem | Yes |
| Runtime | scikit-learn / Python | ONNX Runtime |
| Dynamic batch input | Through Python inference | Yes |
| Current benchmark mean | 0.084 ms | 0.223 ms |
| Current benchmark P95 | 0.101 ms | 0.231 ms |
| Prediction parity tested | Reference | Yes |
| Main advantage | Simple Python-native serialization | Portable inference representation |
| Main limitation | Python-specific and unsafe for untrusted files | Requires export and compatible runtime |

For this project, Pickle remains the reference artifact and ONNX is an
additional portable representation.

---

## 8. Why Keep Both Artifacts?

The two artifacts serve different purposes.

### Pickle

The Pickle artifact preserves the fitted scikit-learn objects used by
the Python prediction implementation:

```text
LinearRegression
+
DictVectorizer
```

This makes it convenient for the existing Python service and for
continuing development within the scikit-learn ecosystem.

### ONNX

The ONNX artifact provides a standardized model representation that can
be consumed by ONNX Runtime and other compatible tooling.

Although ONNX did not provide lower latency in the current benchmark,
it demonstrates that the baseline can be exported and executed outside
the original scikit-learn inference path.

---

## 9. Validation

The serialization implementation was validated at multiple levels.

### Artifact existence

Both artifacts are present:

```text
models/baseline.pkl
models/model.onnx
```

### ONNX Runtime loading

The ONNX model loads successfully and exposes:

```text
input
[None, 4910]
variable
```

### Prediction parity

```text
tests/test_serialization.py::test_onnx_prediction_parity PASSED
```

### Full test suite

The final project test suite contains:

```text
17 tests
```

and all tests pass:

```text
17 passed
```

---

## 10. Final Test and Quality Status

The project currently reports:

```text
17 passed
```

with total test coverage of:

```text
75.99%
```

The configured minimum coverage requirement is:

```text
70%
```

Therefore:

```text
75.99% > 70%
```

The project also passes the configured pre-commit checks:

```text
fix end of files ........ Passed
ruff check .............. Passed
black ................... Passed
```

---

## 11. Integration with the API

The ONNX artifact is part of the model serialization work, while the
current FastAPI prediction service continues to use the Pickle-based
`DurationPredictor`.

The API has already been validated independently with:

```text
GET  /health
GET  /metadata
POST /predict
POST /predict/batch
```

The production API therefore has a validated reference inference path,
while the ONNX artifact provides an additional portable representation
that has been verified for prediction parity.

---

## 12. Reproducibility

The serialization workflow is reproducible from the project artifacts
and source code.

Reference files:

```text
src/prodml/export.py
models/baseline.pkl
models/model.onnx
tests/test_serialization.py
```

The baseline model configuration remains:

```text
Model:          LinearRegression
Features:       PU_DO, trip_distance
Vectorizer:     DictVectorizer
```

---

## 13. Module 4 Conclusion

Module 4 successfully evaluates model serialization by keeping the
existing Pickle artifact and producing an ONNX representation of the
baseline model.

The ONNX artifact:

- was successfully generated
- loads successfully with ONNX Runtime
- exposes the expected dynamic input shape `[None, 4910]`
- produces prediction results consistent with the reference model
- passes the dedicated prediction parity test

The benchmark showed:

```text
Pickle mean latency: 0.084 ms
Pickle p95 latency:  0.101 ms

ONNX mean latency:   0.223 ms
ONNX p95 latency:    0.231 ms
```

For this particular small baseline workload, Pickle/scikit-learn was
faster. Therefore, the main benefit demonstrated by ONNX in this module
is portability and runtime interoperability rather than lower local
inference latency.

---

## 14. Definition of Done

- [x] Existing Pickle model identified
- [x] `LinearRegression` confirmed in Pickle artifact
- [x] `DictVectorizer` confirmed in Pickle artifact
- [x] ONNX model exported
- [x] `models/model.onnx` generated
- [x] Dynamic batch dimension confirmed
- [x] 4,910 input features confirmed
- [x] ONNX Runtime loading verified
- [x] ONNX prediction parity test implemented
- [x] Prediction parity test passed
- [x] Pickle inference benchmark completed
- [x] ONNX Runtime benchmark completed
- [x] Benchmark results documented
- [x] Serialization trade-offs documented
- [x] Full test suite passes
- [x] Coverage requirement passes
- [x] pre-commit checks pass

---

## 15. Module Status

**Module 4 — Complete**

The project now contains both the original Python-native model artifact
and a validated ONNX representation, with automated parity testing and
benchmark results documented for future model-serving decisions.
