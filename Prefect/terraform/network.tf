# Data source for existing default network
data "google_compute_network" "default" {
  name    = "default"
  project = var.project_id
}

# Serverless VPC Access Connector for Cloud Run
resource "google_vpc_access_connector" "prefect_connector" {
  name          = "prefect-connector"
  region        = var.region
  network       = "default"
  ip_cidr_range = var.vpc_connector_ip_range
  project       = var.project_id

  min_instances = 2
  max_instances = 3

  depends_on = [
    google_project_service.vpcaccess,
    google_project_service.compute
  ]
}

# Allocated IP range for Cloud Build private worker pool (Service Networking)
resource "google_compute_global_address" "cloudbuild_peering" {
  name          = "cloudbuild-peering-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 24
  network       = data.google_compute_network.default.id
  project       = var.project_id

  depends_on = [google_project_service.servicenetworking]
}

# Service Networking Connection for Cloud Build (and potentially other services)
resource "google_service_networking_connection" "cloudbuild_peering" {
  network                 = data.google_compute_network.default.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.cloudbuild_peering.name]

  depends_on = [google_project_service.servicenetworking]
}
