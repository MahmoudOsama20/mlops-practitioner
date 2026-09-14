output "mlflow_url" {
  description = "MLflow tracking server URL"
  value       = "http://localhost:5000"
}

output "minio_api_url" {
  description = "MinIO API endpoint"
  value       = "http://localhost:9000"
}

output "minio_console_url" {
  description = "MinIO console URL"
  value       = "http://localhost:9001"
}

output "postgres_host" {
  description = "PostgreSQL host"
  value       = "localhost"
}

output "postgres_port" {
  description = "PostgreSQL port"
  value       = 5432
}
