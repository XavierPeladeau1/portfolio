# Cloud Storage bucket for VM configuration files
resource "google_storage_bucket" "prefect_configs" {
  name     = "${var.project_id}-prefect-configs"
  location = var.region
  project  = var.project_id

  force_destroy = true

  # Enable versioning to track changes
  versioning {
    enabled = true
  }

  # Uniform bucket-level access (recommended)
  uniform_bucket_level_access = true

  labels = merge(var.labels, {
    purpose = "prefect-vm-configs"
  })

  depends_on = [google_project_service.compute]
}

# Docker Compose configuration file
resource "google_storage_bucket_object" "compose_file" {
  name   = "prefect/docker-compose.yml"
  bucket = google_storage_bucket.prefect_configs.name
  content = templatefile("${path.module}/scripts/compose.yml", {
    gcp_project       = var.project_id
    gcp_region        = var.region
    prefect_image     = var.prefect_image
    prefect_server_ip = google_compute_address.prefect_internal.address
  })

  # Update on any change
  lifecycle {
    replace_triggered_by = [
      google_storage_bucket.prefect_configs
    ]
  }
}

# Python script to create Cloud Run work pool
resource "google_storage_bucket_object" "create_pool_script" {
  name   = "prefect/create_cloud_run_pool.py"
  bucket = google_storage_bucket.prefect_configs.name
  source = "${path.module}/scripts/create_cloud_run_pool.py"

}

# Grant the Prefect server service account access to read from the bucket
resource "google_storage_bucket_iam_member" "prefect_server_storage_access" {
  bucket = google_storage_bucket.prefect_configs.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.prefect_server.email}"

  depends_on = [google_service_account.prefect_server]
}
