# Prefect Infrastructure - Terraform

This directory contains Terraform configuration files to provision all GCP infrastructure needed for the Prefect server.

## What This Creates

This Terraform configuration provisions:

1. **Compute Instance**: VM for running Prefect server, database, and worker
2. **Networking**:
   - VPC Access Connector for Cloud Run
   - Private Service Connection for Cloud Build
   - Firewall rules for secure access
3. **IAM**: Service accounts and permissions
4. **Cloud Build**: Private worker pool
5. **Automatic Setup**: Startup script that installs Docker and deploys Prefect

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     GCP Project                         │
│                                                         │
│  ┌──────────────────┐    ┌─────────────────────────┐  │
│  │  Prefect VM      │    │   VPC Access Connector  │  │
│  │  10.162.0.37     │◄───┤   (10.8.0.0/28)         │  │
│  │                  │    │   For Cloud Run         │  │
│  │  - PostgreSQL    │    └─────────────────────────┘  │
│  │  - Prefect API   │                                  │
│  │  - Prefect UI    │    ┌─────────────────────────┐  │
│  │  - Worker        │◄───┤   Cloud Build Pool      │  │
│  └──────────────────┘    │   (Private Network)     │  │
│                          └─────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │         Firewall Rules                           │ │
│  │  - Allow HTTP from VPN/Cloud Run/Cloud Build     │ │
│  │  - Allow SSH (configurable)                      │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Terraform**: Install Terraform >= 1.0
   ```bash
   # macOS
   brew install terraform

   # Or download from https://www.terraform.io/downloads
   ```

2. **GCP CLI**: Install and authenticate with GCP
   ```bash
   # Install gcloud
   # See: https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth application-default login

   # Set project
   gcloud config set project subtle-presence-295420
   ```

3. **Permissions**: Your GCP account needs:
   - Compute Admin
   - Service Account Admin
   - Security Admin (for firewall rules)
   - VPC Access Admin
   - Cloud Build Admin

## Quick Start

### 1. Configure Variables

Copy the example variables file and customize it:

```bash
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars
```

**Important variables to set:**
- `postgres_password`: Set a strong password
- `ssh_allowed_cidrs`: Restrict SSH access (default allows all)
- `project_id`: Your GCP project ID
- `region` and `zone`: Where to deploy resources

### 2. Initialize Terraform

```bash
terraform init
```

This downloads the required providers (Google Cloud).

### 3. Review the Plan

```bash
terraform plan
```

Review what Terraform will create. Verify:
- VM instance configuration
- Network settings
- Firewall rules
- Service accounts

### 4. Apply the Configuration

```bash
terraform apply
```

Type `yes` when prompted. This will:
- Create all infrastructure (~5-10 minutes)
- Start the VM with startup script
- Install Docker on the VM
- Deploy Prefect containers
- Initialize the database

### 5. Verify Deployment

After `terraform apply` completes, it will output important information:

```bash
# View outputs
terraform output

# SSH into the VM
terraform output -raw ssh_command | bash

# Check if services are running
docker compose ps
docker compose logs -f
```

## Configuration Files

### Core Files

- **[main.tf](main.tf)**: Provider configuration and API enablement
- **[variables.tf](variables.tf)**: All configurable variables
- **[outputs.tf](outputs.tf)**: Output values after deployment
- **[terraform.tfvars.example](terraform.tfvars.example)**: Example configuration

### Resource Files

- **[compute.tf](compute.tf)**: VM instance and service account
- **[network.tf](network.tf)**: VPC, subnets, and VPC Access Connector
- **[firewall.tf](firewall.tf)**: Firewall rules
- **[cloudbuild.tf](cloudbuild.tf)**: Cloud Build worker pool

### Scripts

- **[scripts/startup.sh](scripts/startup.sh)**: VM startup script (installs Docker, starts Prefect)

## Managing Infrastructure

### View Current State

```bash
# List all resources
terraform state list

# Show specific resource
terraform state show google_compute_instance.prefect_server

# View outputs
terraform output
```

### Update Infrastructure

After modifying `.tf` files or `terraform.tfvars`:

```bash
# Preview changes
terraform plan

# Apply changes
terraform apply
```

### Destroy Infrastructure

**WARNING**: This will delete all resources!

```bash
terraform destroy
```

## Common Tasks

### SSH into the VM

```bash
# Using Terraform output
$(terraform output -raw ssh_command)

# Or manually
gcloud compute ssh prefect-server-vm \
  --zone=northamerica-northeast1-a \
  --project=subtle-presence-295420
```

### Check Prefect Services

```bash
# SSH into VM first
$(terraform output -raw ssh_command)

# Then check services
cd /opt/prefect-infrastructure
docker compose ps
docker compose logs -f prefect-server
docker compose logs -f prefect-worker
```

### Update Prefect Image

```bash
# SSH into the VM
$(terraform output -raw ssh_command)

# Pull new image
docker pull gcr.io/subtle-presence-295420/prefect-project:latest

# Restart services
cd /opt/prefect-infrastructure
docker compose down
docker compose up -d

# Upgrade database
docker exec prefect-server prefect server database upgrade -y
```

### Access Prefect UI

The Prefect UI is accessible at the internal IP (via VPN):

```bash
# Get the internal IP
terraform output prefect_ui_url_internal

# Output: http://10.162.0.37
```

Connect via OpenVPN, then navigate to this URL.

### Configure Work Pool

After deployment, configure the Cloud Run work pool:

```bash
# SSH into VM
$(terraform output -raw ssh_command)

# Access Prefect server container
docker exec -it prefect-server bash

# Create work pool (if not exists)
prefect work-pool create cloud-run --type cloud-run

# Or configure via UI at http://10.162.0.37
```

Make sure to set the VPC connector in the work pool:
- VPC Connector: `prefect-connector`
- Region: `northamerica-northeast1`

## State Management

### Local State (Default)

By default, Terraform stores state locally in `terraform.tfstate`.

**Important**:
- Don't commit `terraform.tfstate` to git (it's in `.gitignore`)
- Backup this file regularly
- Don't edit manually

### Remote State (Recommended for Teams)

To use GCS for remote state, uncomment in [main.tf](main.tf):

```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "prefect-infrastructure"
  }
}
```

Then initialize:

```bash
# Create bucket
gcloud storage buckets create gs://your-terraform-state-bucket \
  --project=subtle-presence-295420 \
  --location=northamerica-northeast1

# Re-initialize with backend
terraform init -migrate-state
```

## Security Considerations

### 1. SSH Access

By default, SSH is open to `0.0.0.0/0`. Restrict this:

```hcl
# In terraform.tfvars
ssh_allowed_cidrs = ["YOUR_IP/32"]
```

Or use IAP for SSH:

```bash
gcloud compute ssh prefect-server-vm \
  --zone=northamerica-northeast1-a \
  --tunnel-through-iap
```

### 2. Database Password

**Never commit `terraform.tfvars` with passwords!**

Options:
1. Use environment variables:
   ```bash
   export TF_VAR_postgres_password="secure_password"
   terraform apply
   ```

2. Use Google Secret Manager:
   ```hcl
   data "google_secret_manager_secret_version" "db_password" {
     secret = "prefect-db-password"
   }
   ```

3. Input manually:
   ```bash
   terraform apply -var="postgres_password=your_password"
   ```

### 3. Service Account Permissions

The Prefect service account has these roles:
- `roles/run.admin` - Create Cloud Run jobs
- `roles/artifactregistry.reader` - Pull images
- `roles/storage.objectViewer` - Read GCS
- `roles/iam.serviceAccountUser` - Use service accounts
- `roles/logging.logWriter` - Write logs

Review and adjust in [compute.tf](compute.tf) as needed.

## Troubleshooting

### Services Not Starting

Check startup script logs:

```bash
gcloud compute instances get-serial-port-output prefect-server-vm \
  --zone=northamerica-northeast1-a \
  --project=subtle-presence-295420
```

### Cannot Pull Docker Image

Ensure the VM service account has access to GCR:

```bash
# SSH into VM
$(terraform output -raw ssh_command)

# Test authentication
gcloud auth configure-docker
docker pull northamerica-northeast1-docker.pkg.dev/subtle-presence-295420/prefect-project/prefect-project:latest
```

### Cloud Run Jobs Cannot Reach Prefect

1. Verify VPC connector exists:
   ```bash
   terraform output vpc_connector_name
   ```

2. Check firewall allows traffic from `10.8.0.0/28`

3. Verify work pool has VPC connector configured

### Terraform Errors

```bash
# Reset state if corrupted
terraform init -reconfigure

# Refresh state from actual infrastructure
terraform refresh

# Import existing resource
terraform import google_compute_instance.prefect_server prefect-server-vm
```

## Cost Estimation

Approximate monthly costs (us-northeast1):

| Resource | Cost |
|----------|------|
| e2-medium VM (24/7) | ~$25 |
| 50GB Standard Disk | ~$2 |
| VPC Access Connector | ~$9 |
| Cloud Build (minimal use) | ~$1 |
| Networking/Egress | Variable |
| **Total** | **~$37/month** |

Actual costs may vary. Use [GCP Pricing Calculator](https://cloud.google.com/products/calculator) for estimates.

## Outputs Reference

After deployment, Terraform outputs:

| Output | Description |
|--------|-------------|
| `prefect_server_internal_ip` | Internal IP for Prefect server |
| `prefect_server_external_ip` | External IP for internet access |
| `prefect_api_url` | API URL for workers and jobs |
| `prefect_ui_url_internal` | UI URL (via VPN) |
| `vpc_connector_name` | VPC connector for Cloud Run |
| `ssh_command` | Command to SSH into VM |

## Next Steps

After deploying with Terraform:

1. **Verify Services**: SSH into VM and check containers
2. **Configure Work Pool**: Set VPC connector in work pool
3. **Deploy Flows**: Use Cloud Build to deploy flows
4. **Set Up VPN**: Deploy OpenVPN for developer access
5. **Monitor**: Check logs and metrics

## Additional Resources

- [Terraform GCP Provider Docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Prefect Documentation](https://docs.prefect.io/)
- [GCP Compute Engine](https://cloud.google.com/compute/docs)
- [Parent README](../README.md)
- [Firewall Configuration Guide](../gcp-firewall-rules.md)

## Support

For issues:
1. Check Terraform output for errors
2. Review VM startup logs
3. Check [Troubleshooting](#troubleshooting) section
4. Review service logs: `docker compose logs`
