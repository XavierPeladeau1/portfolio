# Firewall rule to allow HTTP traffic to Prefect server from authorized sources
resource "google_compute_firewall" "allow_prefect_server" {
  name    = "allow-prefect-server-access"
  network = "default"
  project = var.project_id

  description = "Allow HTTP traffic to Prefect server from VPN, Cloud Run, and Cloud Build"

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = [
    var.subnet_range,                                                    # Default subnet (VPN access)
    var.vpc_connector_ip_range,                                          # Serverless VPC Access (Cloud Run)
    "${google_compute_global_address.cloudbuild_peering.address}/24",    # Cloud Build private worker pool (Service Networking)
  ]

  target_tags = ["prefect-server"]

  priority = 1000

  depends_on = [google_project_service.compute]
}

# Firewall rule to allow SSH access
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh-prefect-server"
  network = "default"
  project = var.project_id

  description = "Allow SSH access to Prefect server"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.ssh_allowed_cidrs

  target_tags = ["prefect-server"]

  priority = 1000

  depends_on = [google_project_service.compute]
}

# Firewall rule to allow internal communication
resource "google_compute_firewall" "allow_internal" {
  name    = "allow-internal-prefect"
  network = "default"
  project = var.project_id

  description = "Allow internal communication within the VPC"

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.subnet_range]

  priority = 65534

  depends_on = [google_project_service.compute]
}
