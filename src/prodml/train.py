from pathlib import Path

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import prepare_data, vectorize_features
from prodml.logging_conf import configure_logging
from prodml.training import run_experiments


def main() -> None:
    """Run all Module 2 training experiments."""

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------

    data = load_data(
        settings.data_path,
    )

    data = prepare_data(
        data,
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # One split is created and reused by every model family.
    # ------------------------------------------------------------------

    X_train, X_val, y_train, y_val = split_data(
        data,
    )

    # ------------------------------------------------------------------
    # Feature vectorization
    # ------------------------------------------------------------------

    X_train_vectorized, X_val_vectorized, vectorizer = vectorize_features(
        X_train,
        X_val,
    )

    feature_names = vectorizer.feature_names_

    # ------------------------------------------------------------------
    # MLflow experiments
    # ------------------------------------------------------------------

    run_experiments(
        X_train=X_train_vectorized,
        X_val=X_val_vectorized,
        y_train=y_train,
        y_val=y_val,
        feature_names=feature_names,
        data_path=Path(settings.data_path),
        test_size=settings.test_size,
        random_state=settings.random_state,
        artifact_dir=Path("reports") / "module-2-artifacts",
    )


if __name__ == "__main__":
    configure_logging()
    main()
