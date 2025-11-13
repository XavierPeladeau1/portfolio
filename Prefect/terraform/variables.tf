variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
}

variable "zone" {
  description = "GCP zone for VM instance"
  type        = string
}

variable "subnet_range" {
  description = "CIDR range for the default subnet"
  type        = string
}

variable "prefect_server_internal_ip" {
  description = "Internal IP address for Prefect server VM"
  type        = string
}

variable "vpc_connector_ip_range" {
  description = "IP range for Serverless VPC Access connector"
  type        = string
}

variable "prefect_vm_machine_type" {
  description = "Machine type for Prefect server VM"
  type        = string
}

variable "prefect_vm_boot_disk_size" {
  description = "Boot disk size in GB for Prefect server VM"
  type        = number
}

variable "prefect_vm_boot_disk_type" {
  description = "Boot disk type for Prefect server VM"
  type        = string
}

variable "prefect_image" {
  description = "Docker image for Prefect server and worker"
  type        = string
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}

variable "ssh_allowed_cidrs" {
  description = "CIDR ranges allowed to SSH into the Prefect server"
  type        = list(string)
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
}

variable "github_owner" {
  description = "GitHub repository owner/organization"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
}

variable "github_branch" {
  description = "GitHub branch to trigger builds on"
  type        = string
}

