# Prefect Workflow Orchestration Platform on GCP
By: Xavier Péladeau

> A production-ready, self-hosted Prefect orchestration platform with Infrastructure-as-Code provisioning, secure networking, and automated CI/CD deployment on Google Cloud Platform.


## Overview

This project demonstrates the design and implementation of a scalable data workflow orchestration platform using [Prefect](https://www.prefect.io/), deployed on Google Cloud Platform with fully automated infrastructure provisioning through Terraform.

## Technical Skills & Implementation

**Terraform Infrastructure as Code**: Complete GCP infrastructure provisioned with modular configuration files for compute, networking, security, and CI/CD resources

**GCP Cloud Architecture**: VPC networking with private IP addressing, Serverless VPC Access connector for Cloud Run integration, Service Networking for Cloud Build, and zero-trust firewall rules

**Prefect Workflow Orchestration**: Python-based data pipelines with task dependencies, scheduled deployments, and distributed execution on serverless Cloud Run infrastructure

**GitOps CI/CD Pipeline**: Automated Docker image builds and workflow deployment synchronization via Cloud Build, triggered on every commit to main branch

**Docker & Container Orchestration**: Multi-service management with Docker Compose (PostgreSQL, Redis, Prefect server/worker), custom images with workflow dependencies, and serverless execution

**Security Implementation**: Network isolation with no public access to orchestration layer, IAM service accounts with least-privilege permissions, secrets management, and SSH via Identity-Aware Proxy

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        GCP Project                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Prefect Server VM (Docker Compose)                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────┐    │   │
│  │  │ PostgreSQL │  │ Prefect API  │  │  Worker    │    │   │
│  │  │ Database   │  │  & UI        │  │  Process   │    │   │
│  │  └────────────┘  └──────────────┘  └────────────┘    │   │
│  │                  Internal IP: 10.162.0.37:80         │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ▲                                │
│                            │ Secure VPC Access              │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│    ┌────┴────┐      ┌──────┴──────┐    ┌──────┴──────┐      │
│    │ VPC     │      │  Cloud Run  │    │ Cloud Build │      │
│    │Connector│      │  Workflows  │    │   CI/CD     │      │
│    │10.8.0.0 │      │  (Jobs)     │    │  Pipeline   │      │
│    └─────────┘      └─────────────┘    └─────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Firewall: Restrict HTTP to VPC only                 │   │
│  │  - VPN/Subnet: 10.162.0.0/20                         │   │
│  │  - Cloud Run: 10.8.0.0/28                            │   │
│  │  - Cloud Build: Private Service Connection           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

**Prefect Server** (VM-based)
- **Server**: REST API and Web UI for workflow management
- **Database**: PostgreSQL for workflow metadata and execution history
- **Worker**: Polls for scheduled workflows and provisions Cloud Run jobs
- **Redis**: Caching layer for improved performance

**Workflow Execution** (Cloud Run)
- Serverless container execution with auto-scaling
- Custom Docker image with workflow code and dependencies
- Secure VPC connectivity to Prefect server
- Isolated execution environment per workflow run

**Infrastructure** (Terraform-managed)
- Compute Engine VM with automated Docker setup
- VPC networking with subnet isolation
- Firewall rules for security
- Service accounts with least-privilege IAM
- Cloud Build for CI/CD automation


## Project Structure

```
.
├── terraform/                 # Infrastructure as Code
│   ├── main.tf                # Provider and API configuration
│   ├── compute.tf             # VM instance and service accounts
│   ├── network.tf             # VPC and networking
│   ├── firewall.tf            # Security rules
│   ├── cloudbuild.tf          # CI/CD infrastructure
│   ├── variables.tf           # Input variables
│   ├── outputs.tf             # Output values
│   └── scripts/
│       └── startup.sh         # VM initialization script
├── flows/                     # Workflow definitions
│   ├── dbt_build.py           # Data transformation workflows
│   └── test_flow.py           # Testing workflow
├── Dockerfile                 # Custom Docker image for workflows
├── cloudbuild.yaml            # CI/CD pipeline configuration
├── prefect.yaml               # Prefect deployment configuration
└── requirements.txt           # Python dependencies for custom image
```

## Getting Started

### Prerequisites

- Google Cloud Platform account
- Terraform >= 1.0
- `gcloud` CLI authenticated
- Appropriate GCP IAM permissions

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd prefect-project
   ```

2. **Configure Terraform variables**
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your configuration
   ```

3. **Provision infrastructure**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

4. **Access Prefect UI**

   Examine the Terraform output to find commands for accessing the Prefect UI via SSH tunnel, then open `http://localhost:4200` in your browser.

For detailed setup instructions, see [terraform/README.md](terraform/README.md).

## Example Workflows

### dbt Data Transformation
Orchestrates dbt model builds with configurable targets and selectors:
```python
@flow
def dbt_build(selector: str, target: str, profile: str):
    dbt_runner.test_models(selector, target, profile)
    dbt_runner.run_models(selector, target, profile)
```

## Operational Tasks

### Deploy New Workflows (Automated CI/CD)

The project uses an automated CI/CD pipeline via Cloud Build. Deployments are automatically synced on every commit to the main branch.

**Workflow:**
1. **Define or modify workflows** in the `flows/` directory
2. **Update deployment configuration** in [prefect.yaml](prefect.yaml)
3. **Commit and push** to the main branch

**What happens automatically:**
1. Cloud Build triggers on push to main
2. Builds new Docker image with updated code
3. Pushes image to Artifact Registry
4. Runs `prefect deploy --all` to sync deployments with Prefect server
5. New/updated workflows are immediately available