"""
Script to create a Google Cloud Run work pool in Prefect.

This script creates a Cloud Run worker pool that can be used to execute
Prefect flows on Google Cloud Run.

Environment Variables:
    POOL_NAME: Name of the work pool to create (default: cloud-run-pool)
    GCP_PROJECT: GCP project ID
    GCP_REGION: GCP region for Cloud Run deployment (default: us-central1)
    VPC_CONNECTOR_NAME: VPC connector name for private network access
    PREFECT_IMAGE: Container image name to use for the Cloud Run worker
"""

import asyncio
import os
from prefect import get_client
from prefect.client.schemas.actions import WorkPoolCreate

# Import cloud run v2 worker to get the default base job template
from prefect_gcp.workers.cloud_run_v2 import CloudRunWorkerV2


async def create_cloud_run_pool(
    pool_name: str,
    project_id: str,
    region: str,
    vpc_connector_name: str,
    image_name: str
):
    """
    Create a Cloud Run work pool.

    Args:
        pool_name: Name of the work pool to create
        project_id: GCP project ID (optional, can be set in pool config later)
        region: GCP region for Cloud Run deployment
        vpc_connector_name: VPC connector name for accessing private resources (e.g., 'prefect-connector')
    """
    async with get_client() as client:
        # Check if work pool already exists
        try:
            existing_pool = await client.read_work_pool(work_pool_name=pool_name)

            print(f"Work pool '{pool_name}' already exists.")
            return existing_pool
        except Exception:
            # Pool doesn't exist, create it
            pass

        # Get the default base job template from CloudRunWorkerV2
        base_job_template = CloudRunWorkerV2.get_default_base_job_template()

        # Customize the template with our region and project settings
        base_job_template["job_configuration"]["region"] = region
        base_job_template["variables"]["properties"]["region"]["default"] = region

        base_job_template["job_configuration"]["project"] = project_id

        # Configure VPC connector for private network access
        # Full VPC connector path format: projects/{project}/locations/{region}/connectors/{name}
        vpc_connector_path = f"projects/{project_id}/locations/{region}/connectors/{vpc_connector_name}"
        base_job_template["variables"]["properties"]["vpc_connector_name"]["default"] = vpc_connector_path

        # Configure default image
        base_job_template["variables"]["properties"]["image"]["default"] = image_name

        # Create the work pool
        work_pool = WorkPoolCreate(
            name=pool_name,
            type="cloud-run-v2",
            base_job_template=base_job_template,
            description=f"Cloud Run work pool for executing flows in GCP region {region}",
        )

        created_pool = await client.create_work_pool(work_pool=work_pool)
        print(f"Successfully created Cloud Run work pool: '{pool_name}'")

        return created_pool


async def main():
    """Main function to create the work pool."""
    # Get configuration from environment variables
    pool_name = os.getenv("POOL_NAME")
    project_id = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    vpc_connector_name = os.getenv("VPC_CONNECTOR_NAME")
    image_name = os.getenv("PREFECT_IMAGE")

    await create_cloud_run_pool(
        pool_name=pool_name,
        project_id=project_id,
        region=region,
        vpc_connector_name=vpc_connector_name,
        image_name=image_name,
    )


if __name__ == "__main__":
    asyncio.run(main())
