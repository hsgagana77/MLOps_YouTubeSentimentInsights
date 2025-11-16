#!/bin/bash
set -e

echo "=============================="
echo " MLflow Server Setup Script"
echo "=============================="

# --- Step 1: System update ---
echo " Updating system packages..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv pipenv virtualenv

# --- Step 2: Create working directory ---
echo " Creating mlflow directory..."
mkdir -p ~/mlflow
cd ~/mlflow

# --- Step 3: Setup virtual environment with Pipenv ---
echo " Setting up Pipenv environment..."
pipenv install mlflow awscli boto3

# --- Step 4: Activate virtual environment ---
echo " Activating virtual environment..."
VENV_PATH=$(pipenv --venv)
source "$VENV_PATH/bin/activate"

# --- Step 5: AWS Configuration ---
echo " Configuring AWS credentials..."
read -p "Enter AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -s -p "Enter AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo
read -p "Enter AWS Region (e.g., ap-south-1): " AWS_DEFAULT_REGION

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION

# Persist credentials using AWS CLI
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set default.region "$AWS_DEFAULT_REGION"

# --- Step 6: Launch MLflow server ---
echo " Starting MLflow server..."
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://mlflow-bucket-9779 \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts '*' > mlflow.log 2>&1 &

echo " MLflow server started successfully!"
echo " Logs: $(pwd)/mlflow.log"
echo " Access UI at: http://<your-server-ip>:5000"