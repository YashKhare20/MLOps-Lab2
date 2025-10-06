#!/bin/bash
set -e

echo "Starting Airflow Lab 2 Setup..."

# Remove existing .env file if it exists
rm -f .env
rm -rf ./logs ./plugins ./working_data ./model

# Stop and remove containers, networks, and volumes
echo "Cleaning up existing Docker containers..."
docker compose down -v

# Create required Airflow directories
echo "Creating required directories..."
mkdir -p ./logs ./plugins ./working_data ./model

# Write the current user's UID into .env
echo "AIRFLOW_UID=$(id -u)" > .env
echo "Created .env file with AIRFLOW_UID=$(id -u)"

# Run airflow CLI to show current config
docker compose run --rm airflow-cli airflow config list
