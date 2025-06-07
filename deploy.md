# Google Cloud Run Deployment Guide

This guide will help you deploy the LLM Validation Framework to Google Cloud Run.

## Prerequisites

1. **Google Cloud Project**: Ensure you have a Google Cloud project with billing enabled
2. **Google Cloud CLI**: Install and configure the `gcloud` CLI tool
3. **Docker**: Install Docker on your local machine
4. **API Keys**: Obtain API keys for:
   - OpenAI (required)
   - Anthropic (required)
   - Voyage AI (optional, for embeddings)
   - DeepSeek (optional)

## Setup Instructions

### 1. Enable Required APIs

```bash
# Enable required Google Cloud APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Set Environment Variables

```bash
# Set your project ID
export PROJECT_ID="your-google-cloud-project-id"
gcloud config set project $PROJECT_ID
```

### 3. Configure API Keys

You have two options for setting API keys:

#### Option A: Using Cloud Build Substitutions (Recommended)

Update the substitutions in `cloudbuild.yaml`:

```yaml
substitutions:
  _OPENAI_API_KEY: 'your-actual-openai-api-key'
  _ANTHROPIC_API_KEY: 'your-actual-anthropic-api-key'
  _VOYAGE_API_KEY: 'your-actual-voyage-api-key'
  _DEEPSEEK_API_KEY: 'your-actual-deepseek-api-key'
```

#### Option B: Using Google Secret Manager (More Secure)

```bash
# Create secrets
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-anthropic-api-key" | gcloud secrets create anthropic-api-key --data-file=-
echo -n "your-voyage-api-key" | gcloud secrets create voyage-api-key --data-file=-
echo -n "your-deepseek-api-key" | gcloud secrets create deepseek-api-key --data-file=-

# Grant Cloud Run access to secrets
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_ID@appspot.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 4. Deploy Using Cloud Build

```bash
# Submit build to Cloud Build
gcloud builds submit --config cloudbuild.yaml .
```

### 5. Manual Deployment (Alternative)

If you prefer to build and deploy manually:

```bash
# Build the Docker image
docker build -t gcr.io/$PROJECT_ID/llm-validation-app .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/llm-validation-app

# Deploy to Cloud Run
gcloud run deploy llm-validation-app \
  --image gcr.io/$PROJECT_ID/llm-validation-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 10 \
  --max-instances 5 \
  --set-env-vars OPENAI_API_KEY="your-openai-key",ANTHROPIC_API_KEY="your-anthropic-key"
```

## Configuration Options

### Resource Allocation

The current configuration allocates:
- **Memory**: 4GB (adjust based on your data size)
- **CPU**: 2 vCPUs
- **Timeout**: 1 hour (for long-running validations)
- **Concurrency**: 10 requests per instance
- **Max Instances**: 5 (adjust based on expected load)

### Environment Variables

The following environment variables are automatically set:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `VOYAGE_API_KEY`: Your Voyage AI API key (optional)
- `DEEPSEEK_API_KEY`: Your DeepSeek API key (optional)
- `PYTHONUNBUFFERED`: Set to 1 for proper logging

## Data and Output Management

### Included Data

The Docker image includes:
- **Guidelines**: All PDF files in `data/Guidelines/` (RECORD, STROBE, Li-Paper, CHEERS)
- **Papers**: All research papers in `data/Papers/` for validation
- **Validation Data**: Reference data in `data/validation/`

### Output Storage

- Results are stored in the container's `/app/output` directory
- For persistent storage, consider mounting a Google Cloud Storage bucket
- Download results through the Streamlit interface

### Persistent Storage (Optional)

To add persistent storage using Google Cloud Storage:

```bash
# Create a storage bucket
gsutil mb gs://$PROJECT_ID-llm-validation-output

# Update Cloud Run service to mount the bucket
gcloud run services update llm-validation-app \
  --region us-central1 \
  --add-volume name=output-volume,type=cloud-storage,bucket=$PROJECT_ID-llm-validation-output \
  --add-volume-mount volume=output-volume,mount-path=/app/output
```

## Monitoring and Logging

### View Logs

```bash
# View Cloud Run logs
gcloud logs read --service=llm-validation-app --region=us-central1

# Follow logs in real-time
gcloud logs tail --service=llm-validation-app --region=us-central1
```

### Monitoring

- Access Cloud Run metrics in the Google Cloud Console
- Set up alerts for high memory usage or errors
- Monitor API usage and costs

## Security Considerations

1. **API Keys**: Use Google Secret Manager for production deployments
2. **Authentication**: Consider enabling authentication for production use
3. **Network**: Configure VPC if needed for additional security
4. **IAM**: Follow principle of least privilege for service accounts

## Troubleshooting

### Common Issues

1. **Build Timeout**: Increase timeout in `cloudbuild.yaml` if needed
2. **Memory Issues**: Increase memory allocation for large datasets
3. **API Rate Limits**: Monitor API usage and implement rate limiting
4. **Cold Starts**: Consider using minimum instances for better performance

### Debug Commands

```bash
# Check service status
gcloud run services describe llm-validation-app --region us-central1

# View recent deployments
gcloud run revisions list --service llm-validation-app --region us-central1

# Test the deployed service
curl -X GET https://your-service-url.run.app/_stcore/health
```

## Cost Optimization

1. **Auto-scaling**: Use minimum instances of 0 for cost savings
2. **Resource Allocation**: Right-size CPU and memory based on usage
3. **Regional Deployment**: Choose the most cost-effective region
4. **API Usage**: Monitor and optimize LLM API calls

## Next Steps

After successful deployment:

1. Access your application at the provided Cloud Run URL
2. Test the validation pipeline with sample papers
3. Set up monitoring and alerting
4. Configure backup and disaster recovery if needed
5. Consider setting up CI/CD for automated deployments

## Support

For issues with deployment:
1. Check Cloud Build logs for build errors
2. Review Cloud Run logs for runtime issues
3. Verify API key configuration
4. Ensure all required APIs are enabled
