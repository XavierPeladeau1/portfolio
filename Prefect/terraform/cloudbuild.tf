# Cloud Build Worker Pool for private connectivity
resource "google_cloudbuild_worker_pool" "private_pool" {
  name     = "my-private-pool"
  location = var.region
  project  = var.project_id

  worker_config {
    disk_size_gb   = 100
    machine_type   = "e2-medium"
    no_external_ip = false
  }

  network_config {
    peered_network = data.google_compute_network.default.id
  }

  depends_on = [
    google_project_service.cloudbuild,
    google_service_networking_connection.cloudbuild_peering
  ]
}

# Grant Cloud Build service account permissions
resource "google_project_iam_member" "cloudbuild_sa_gcr" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_sa_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "prefect_project" {
  location      = var.region
  repository_id = "prefect-project"
  description   = "Docker repository for Prefect project"
  format        = "DOCKER"
  project       = var.project_id

  depends_on = [
    google_project_service.artifact_registry
  ]
}


# Grant Cloud Build service account permissions to push to Artifact Registry
resource "google_project_iam_member" "cloudbuild_sa_artifact_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_cloudbuildv2_connection" "github_connection" {
  location = var.region
  project  = var.project_id
  name     = "github-connection"

  github_config {
    app_installation_id = 91461285

    authorizer_credential {
      oauth_token_secret_version = "projects/${var.project_id}/secrets/github-connection-github-oauthtoken-bf4068/versions/latest"
    }
  }
}

resource "google_cloudbuildv2_repository" "prefect_project_repo" {
  project           = var.project_id
  location          = var.region
  name              = "prefect-project-repo"
  parent_connection = google_cloudbuildv2_connection.github_connection.name
  remote_uri        = "https://github.com/${var.github_owner}/${var.github_repo}.git"
}

# Cloud Build trigger for Docker image builds
# resource "google_cloudbuild_trigger" "prefect_project_build" {
#   name        = "prefect-project-build"
#   project     = var.project_id

#   repository_event_config {
#     repository = google_cloudbuildv2_repository.prefect_project_repo.id
#     push {
#       branch = "^${var.github_branch}$"
#     }
#   }

#   filename = "cloudbuild.yaml"


#   depends_on = [
#     google_cloudbuild_worker_pool.private_pool,
#     google_cloudbuildv2_repository.prefect_project_repo
#   ]
# }

# Data source to get project number
data "google_project" "project" {
  project_id = var.project_id
}
