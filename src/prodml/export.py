import pickle
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings


def export_to_onnx(
    model_path: Path,
    onnx_path: Path,
) -> None:
    """Export the fitted LinearRegression model to ONNX."""

    with open(model_path, "rb") as file:
        artifact = pickle.load(file)

    model = artifact["model"]

    n_features = model.n_features_in_

    initial_types = [
        (
            "input",
            FloatTensorType([None, n_features]),
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
    )

    onnx_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(onnx_path, "wb") as file:
        file.write(onnx_model.SerializeToString())


def main() -> None:
    """Export the baseline model to ONNX."""

    export_to_onnx(
        settings.model_path,
        settings.onnx_path,
    )


if __name__ == "__main__":
    main()
