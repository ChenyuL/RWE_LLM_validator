#!/bin/bash

# Google Cloud Run Deployment Script for LLM Validation Framework
# Usage: ./deploy.sh [PROJECT_ID]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install it first."
    exit 1
fi

# Get project ID
if [ -z "$1" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        print_error "No project ID provided and no default project set."
        echo "Usage: $0 [PROJECT_ID]"
        exit 1
    fi
    print_warning "Using default project: $PROJECT_ID"
else
    PROJECT_ID=$1
    print_status "Using project: $PROJECT_ID"
fi

# Set the project
print_status "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Check if required APIs are enabled
print_status "Checking required APIs..."
REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_success "$api is enabled"
    else
        print_status "Enabling $api..."
        gcloud services enable $api
    fi
done

# Check if API keys are configured
print_status "Checking API key configuration..."
if [ ! -f ".env" ]; then
    print_warning "No .env file found. Make sure to configure API keys in cloudbuild.yaml"
else
    print_success ".env file found"
fi

# Build and deploy options
echo ""
echo "Choose deployment method:"
echo "1. Cloud Build (Recommended)"
echo "2. Local build and deploy"
echo "3. Just build locally (no deploy)"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        print_status "Deploying using Cloud Build..."
        
        # Check if cloudbuild.yaml exists
        if [ ! -f "cloudbuild.yaml" ]; then
            print_error "cloudbuild.yaml not found!"
            exit 1
        fi
        
        # Submit to Cloud Build
        print_status "Submitting build to Cloud Build..."
        gcloud builds submit --config cloudbuild.yaml .
        
        if [ $? -eq 0 ]; then
            print_success "Deployment completed successfully!"
            
            # Get the service URL
            SERVICE_URL=$(gcloud run services describe llm-validation-app --region=us-central1 --format="value(status.url)" 2>/dev/null)
            if [ ! -z "$SERVICE_URL" ]; then
                print_success "Your application is available at: $SERVICE_URL"
            fi
        else
            print_error "Deployment failed!"
            exit 1
        fi
        ;;
        
    2)
        print_status "Building and deploying locally..."
        
        # Build the image
        IMAGE_NAME="gcr.io/$PROJECT_ID/llm-validation-app"
        print_status "Building Docker image: $IMAGE_NAME"
        docker build -t $IMAGE_NAME .
        
        # Push the image
        print_status "Pushing image to Google Container Registry..."
        docker push $IMAGE_NAME
        
        # Deploy to Cloud Run
        print_status "Deploying to Cloud Run..."
        gcloud run deploy llm-validation-app \
            --image $IMAGE_NAME \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated \
            --memory 4Gi \
            --cpu 2 \
            --timeout 3600 \
            --concurrency 10 \
            --max-instances 5
        
        if [ $? -eq 0 ]; then
            print_success "Deployment completed successfully!"
            
            # Get the service URL
            SERVICE_URL=$(gcloud run services describe llm-validation-app --region=us-central1 --format="value(status.url)")
            print_success "Your application is available at: $SERVICE_URL"
        else
            print_error "Deployment failed!"
            exit 1
        fi
        ;;
        
    3)
        print_status "Building Docker image locally..."
        
        IMAGE_NAME="gcr.io/$PROJECT_ID/llm-validation-app"
        docker build -t $IMAGE_NAME .
        
        if [ $? -eq 0 ]; then
            print_success "Image built successfully: $IMAGE_NAME"
            print_status "To deploy later, run:"
            echo "  docker push $IMAGE_NAME"
            echo "  gcloud run deploy llm-validation-app --image $IMAGE_NAME --region us-central1"
        else
            print_error "Build failed!"
            exit 1
        fi
        ;;
        
    *)
        print_error "Invalid choice!"
        exit 1
        ;;
esac

print_success "Script completed!"

# Show useful commands
echo ""
print_status "Useful commands:"
echo "  View logs: gcloud logs tail --service=llm-validation-app --region=us-central1"
echo "  Update service: gcloud run services update llm-validation-app --region=us-central1"
echo "  Delete service: gcloud run services delete llm-validation-app --region=us-central1"
