# VM Instance Information
output "prefect_server_internal_ip" {
  description = "Internal IP address of the Prefect server VM"
  value       = google_compute_instance.prefect_server.network_interface[0].network_ip
}

output "prefect_server_external_ip" {
  description = "External IP address of the Prefect server VM"
  value       = google_compute_instance.prefect_server.network_interface[0].access_config[0].nat_ip
}

output "prefect_server_name" {
  description = "Name of the Prefect server VM instance"
  value       = google_compute_instance.prefect_server.name
}

output "prefect_server_zone" {
  description = "Zone where the Prefect server VM is deployed"
  value       = google_compute_instance.prefect_server.zone
}

# Prefect URLs
output "prefect_api_url" {
  description = "Prefect API URL for worker and job configuration"
  value       = "http://${google_compute_instance.prefect_server.network_interface[0].network_ip}/api"
}

output "prefect_ui_url_internal" {
  description = "Prefect UI URL (accessible via VPN)"
  value       = "http://${google_compute_instance.prefect_server.network_interface[0].network_ip}"
}

# Debugging Commands
output "ssh_command" {
  description = "Command to SSH into the Prefect server"
  value       = "gcloud compute ssh ${google_compute_instance.prefect_server.name} --zone=${google_compute_instance.prefect_server.zone} --project=${var.project_id}"
}

output "ssh_tunnel_iap_command" {
  description = "Command to create SSH tunnel through IAP to access Prefect UI on localhost:4200"
  value       = "gcloud compute ssh ${google_compute_instance.prefect_server.name} --zone=${google_compute_instance.prefect_server.zone} --tunnel-through-iap --project=${var.project_id} -- -L 4200:localhost:80 -N"
}

output "startup_script_logs_command" {
  description = "Command to view startup script logs from serial console"
  value       = "gcloud compute instances get-serial-port-output ${google_compute_instance.prefect_server.name} --zone=${google_compute_instance.prefect_server.zone} --project=${var.project_id}"
}

output "startup_script_logs_tail_command" {
  description = "Command to tail startup script logs in real-time"
  value       = "gcloud compute instances tail-serial-port-output ${google_compute_instance.prefect_server.name} --zone=${google_compute_instance.prefect_server.zone} --project=${var.project_id}"
}

output "startup_script_ssh_logs_command" {
  description = "Command to view startup script logs via SSH"
  value       = "gcloud compute ssh ${google_compute_instance.prefect_server.name} --zone=${google_compute_instance.prefect_server.zone} --project=${var.project_id} --command='sudo journalctl -u google-startup-scripts.service'"
}

output "artifact_registry_image_url" {
  description = "Artifact Registry Docker image URL for Prefect Cloud Run workers"
  value       = "${google_artifact_registry_repository.prefect_project.registry_uri}/prefect-project:latest"
}