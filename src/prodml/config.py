from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    data_path: Path = PROJECT_ROOT / "datasets" / "green_tripdata_2026-01.csv"
    model_path: Path = PROJECT_ROOT / "models" / "baseline.pkl"
    onnx_path: Path = PROJECT_ROOT / "models" / "model.onnx"
    report_path: Path = PROJECT_ROOT / "reports" / "module-1.md"

    test_size: float = 0.2
    random_state: int = 42

    model_version: str = "0.1.0"
    training_date: str = "2026-08-30"
    framework: str = "scikit-learn"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PRODML_",
        extra="ignore",
    )


settings = Settings()
