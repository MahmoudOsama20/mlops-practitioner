resource "docker_network" "mlops" {
  name = "prodml-network"
}

resource "docker_container" "postgres" {
  name  = "prodml-postgres"
  image = docker_image.postgres.image_id

  networks_advanced {
    name    = docker_network.mlops.name
    aliases = ["postgres"]
  }

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.postgres_db}",
  ]

  ports {
    internal = 5432
    external = 5432
  }

  volumes {
    volume_name    = var.postgres_volume_name
    container_path = "/var/lib/postgresql/data"
  }

  restart = "unless-stopped"
}

resource "docker_container" "minio" {
  name  = "prodml-minio"
  image = docker_image.minio.image_id

  networks_advanced {
    name    = docker_network.mlops.name
    aliases = ["minio"]
  }

  command = [
    "server",
    "/data",
    "--console-address",
    ":9001",
  ]

  env = [
    "MINIO_ROOT_USER=${var.minio_root_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_root_password}",
  ]

  ports {
    internal = 9000
    external = 9000
  }

  ports {
    internal = 9001
    external = 9001
  }

  volumes {
    volume_name    = var.minio_volume_name
    container_path = "/data"
  }

  restart = "unless-stopped"
}

resource "docker_container" "mlflow" {
  name  = "prodml-mlflow"
  image = docker_image.mlflow.image_id

  networks_advanced {
    name = docker_network.mlops.name
  }

  depends_on = [
    docker_container.postgres,
    docker_container.minio,
  ]

  env = [
    "AWS_ACCESS_KEY_ID=${var.minio_root_user}",
    "AWS_SECRET_ACCESS_KEY=${var.minio_root_password}",
    "MLFLOW_S3_ENDPOINT_URL=http://minio:9000",
  ]

  ports {
    internal = 5000
    external = 5000
  }

  command = [
    "mlflow",
    "server",
    "--host",
    "0.0.0.0",
    "--port",
    "5000",
    "--backend-store-uri",
    "postgresql://${var.postgres_user}:${var.postgres_password}@postgres:5432/${var.postgres_db}",
    "--artifacts-destination",
    "s3://${var.mlflow_bucket}",
    "--serve-artifacts",
  ]

  restart = "unless-stopped"
}

resource "docker_image" "postgres" {
  name         = var.postgres_image
  keep_locally = true
}

resource "docker_image" "minio" {
  name         = var.minio_image
  keep_locally = true
}

resource "docker_image" "mlflow" {
  name         = var.mlflow_image
  keep_locally = true
}
