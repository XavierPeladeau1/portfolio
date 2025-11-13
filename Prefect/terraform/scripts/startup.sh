#!/bin/bash
set -e

# Startup script for Prefect server VM
echo "Starting Prefect server VM initialization..."

# Update and install dependencies
apt-get update
apt-get install -y curl git

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose plugin
apt-get update
apt-get install -y docker-compose-plugin

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Create directory for Prefect infrastructure
mkdir -p /opt/prefect-infrastructure
cd /opt/prefect-infrastructure

# Download all configuration files from Cloud Storage
echo "Downloading configuration files from gs://${config_bucket}/prefect/..."
gsutil -m cp -r gs://${config_bucket}/prefect/* /opt/prefect-infrastructure/

if [ $? -ne 0 ]; then
  echo "ERROR: Failed to download files from Cloud Storage"
  exit 1
fi

echo "Successfully downloaded all configuration files"

# Start all services
echo "Starting all Prefect services..."
cd /opt/prefect-infrastructure
docker compose up -d || echo "Failed to start services - check logs"

# Wait for services to be healthy
echo "Waiting for services to become healthy..."
sleep 60

# Initialize database
echo "Initializing Prefect database..."
docker exec prefect-infrastructure-prefect-server-1 prefect server database upgrade -y || echo "Database upgrade failed - may already be initialized"

echo "Prefect server VM initialization complete!"
