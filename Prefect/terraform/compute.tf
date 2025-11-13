# Static internal IP address for Prefect server
resource "google_compute_address" "prefect_internal" {
  name         = "prefect-server-internal-ip"
  address_type = "INTERNAL"
  address      = var.prefect_server_internal_ip
  region       = var.region
  project      = var.project_id

  depends_on = [google_project_service.compute]
}

# Prefect Server VM Instance
resource "google_compute_instance" "prefect_server" {
  name         = "prefect-server-vm"
  machine_type = var.prefect_vm_machine_type
  zone         = var.zone
  project      = var.project_id

  tags = ["prefect-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.prefect_vm_boot_disk_size
      type  = var.prefect_vm_boot_disk_type
    }
  }

  network_interface {
    network    = "default"
    subnetwork = null
    network_ip = google_compute_address.prefect_internal.address

    # External IP for internet access
    access_config {
      # Ephemeral external IP
    }
  }

  # Startup script to prepare the VM
  metadata_startup_script = templatefile("${path.module}/scripts/startup.sh", {
    config_bucket = google_storage_bucket.prefect_configs.name
    prefect_image = var.prefect_image
    project_id    = var.project_id
    region        = var.region
  })

  # Service account with necessary permissions
  service_account {
    email  = google_service_account.prefect_server.email
    scopes = ["cloud-platform"]
  }

  labels = var.labels

  allow_stopping_for_update = true

  depends_on = [
    google_project_service.compute,
    google_compute_firewall.allow_prefect_server,
    google_vpc_access_connector.prefect_connector,
    google_storage_bucket.prefect_configs,
    google_storage_bucket_object.compose_file,
    google_storage_bucket_object.create_pool_script
  ]
}

# Service account for Prefect server
resource "google_service_account" "prefect_server" {
  account_id   = "prefect-server-sa"
  display_name = "Prefect Server Service Account"
  project      = var.project_id
}

# IAM permissions for the Prefect server service account
resource "google_project_iam_member" "prefect_cloud_run" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.prefect_server.email}"
}

resource "google_project_iam_member" "prefect_artifact_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.prefect_server.email}"
}

resource "google_project_iam_member" "prefect_storage_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.prefect_server.email}"
}

resource "google_project_iam_member" "prefect_service_account_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.prefect_server.email}"
}

resource "google_project_iam_member" "prefect_logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.prefect_server.email}"
}
