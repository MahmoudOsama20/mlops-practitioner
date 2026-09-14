variable "docker_host" {
  description = "Docker daemon endpoint"
  type        = string
  default     = "npipe:////./pipe/docker_engine"
}

variable "postgres_image" {
  description = "PostgreSQL Docker image"
  type        = string
  default     = "postgres:16"
}

variable "minio_image" {
  description = "MinIO Docker image"
  type        = string
  default     = "quay.io/minio/minio:latest"
}

variable "mlflow_image" {
  description = "MLflow Docker image"
  type        = string
  default     = "ghcr.io/mlflow/mlflow:latest"
}

variable "postgres_user" {
  description = "MLflow PostgreSQL username"
  type        = string
  default     = "mlflow"
}

variable "postgres_password" {
  description = "MLflow PostgreSQL password"
  type        = string
  sensitive   = true
  default     = "mlflow"
}

variable "postgres_db" {
  description = "MLflow PostgreSQL database"
  type        = string
  default     = "mlflow"
}

variable "minio_root_user" {
  description = "MinIO root username"
  type        = string
  default     = "minio"
}

variable "minio_root_password" {
  description = "MinIO root password"
  type        = string
  sensitive   = true
  default     = "minio123"
}

variable "mlflow_bucket" {
  description = "S3 bucket used by MLflow"
  type        = string
  default     = "mlflow"
}

variable "postgres_volume_name" {
  description = "Existing persistent Docker volume for PostgreSQL"
  type        = string
  default     = "docker_postgres_data"
}

variable "minio_volume_name" {
  description = "Existing persistent Docker volume for MinIO"
  type        = string
  default     = "docker_minio_data"
}
