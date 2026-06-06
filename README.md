# Earnings Extraction Agent

Extracts structured trading signals from earnings call transcripts using LLMs.
Built with Anthropic Claude, FastAPI, and deployed on GCP Cloud Run.

## Architecture

```
Transcript (raw text)
        ↓
  FastAPI /extract endpoint
        ↓
  EarningsExtractionAgent (Claude claude-sonnet-4-20250514)
        ↓
  Pydantic validation + confidence scoring
        ↓
  Structured JSON → GCS (gs://your-bucket/extractions/{ticker}/{timestamp}.json)
        ↓
  ExtractionResponse (trustworthy flag, flagged fields, clean signals)
```

## Signals Extracted

| Signal | Description |
|--------|-------------|
| `revenue_actual` | Reported revenue this quarter |
| `eps_actual` | Reported EPS this quarter |
| `revenue_guidance` | Forward revenue guidance |
| `eps_guidance` | Forward EPS guidance |
| `management_tone` | bullish / neutral / bearish |
| `key_risks` | Top 3 risks mentioned |
| `key_opportunities` | Top 3 opportunities mentioned |
| `guidance_raised` | Boolean — did they raise guidance? |
| `buyback_announced` | Boolean — share buyback announced? |
| `dividend_change` | increase / decrease / none |

Each signal includes a `confidence` score (0.0–1.0) and the supporting quote.
Fields below 0.5 confidence are flagged and excluded from the trading payload.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your_key_here
export GCS_BUCKET_NAME=your_bucket_name  # optional for local dev

# Run the server
uvicorn app.main:app --reload --port 8080

# Run tests
pytest tests/ -v
```

## API Endpoints

### POST /extract
Submit a transcript for extraction.

```bash
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "transcript": "Good morning. Revenue was $94.8 billion...",
    "source": "Q3 2025 earnings call"
  }'
```

Response:
```json
{
  "ticker": "AAPL",
  "trustworthy": true,
  "overall_confidence": 0.96,
  "low_confidence_fields": [],
  "signals": {
    "revenue_actual": {"value": "$94.8 billion", "confidence": 1.0},
    "management_tone": {"value": "bullish", "confidence": 0.9}
  },
  "message": "Extraction complete. Overall confidence: 96%."
}
```

### GET /extractions/{ticker}/latest
Retrieve the most recent extraction for a ticker.

### GET /extractions/{ticker}
List all stored extractions for a ticker.

### GET /health
Health check.

## GCP Setup

### 1. Create GCS bucket
```bash
gsutil mb -l us-central1 gs://your-earnings-agent-bucket
```

### 2. Create Artifact Registry repository
```bash
gcloud artifacts repositories create earnings-agent \
  --repository-format=docker \
  --location=us-central1
```

### 3. Store Anthropic API key in Secret Manager
```bash
echo -n "your_anthropic_key" | \
  gcloud secrets create anthropic-api-key --data-file=-
```

### 4. GitHub Actions secrets required
Set these in your GitHub repo Settings → Secrets:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Workload identity provider resource name |
| `GCP_SERVICE_ACCOUNT` | Service account email for deployments |
| `GCS_BUCKET_NAME` | GCS bucket name for storing extractions |

### 5. Workload Identity Federation (keyless auth)
```bash
# Create service account
gcloud iam service-accounts create earnings-agent-sa \
  --display-name="Earnings Agent Deploy SA"

# Grant roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:earnings-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:earnings-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:earnings-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:earnings-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Set up Workload Identity Federation
# (follow: https://github.com/google-github-actions/auth#workload-identity-federation)
```

## CI/CD

On every push to `main`:
1. Tests run (`pytest`)
2. If tests pass, Docker image built and pushed to Artifact Registry
3. Cloud Run service updated with new image
4. Secrets injected from Secret Manager at runtime (never in image)

Pull requests run tests only — no deployment.

## Design Decisions

**Confidence scoring:** Every extracted field gets a 0.0–1.0 confidence score.
Fields below 0.5 are flagged and excluded from `to_trading_payload()`. The
`trustworthy` boolean gates whether the full extraction should be fed downstream.
This prevents low-quality extractions from reaching trading systems.

**Transcript truncation:** Transcripts are truncated to 12,000 characters before
sending to the LLM to stay within context limits while capturing the most
relevant parts (typically the first portion contains prepared remarks with the
key numbers).

**Structured output:** The LLM is prompted to return JSON only. The agent strips
markdown fences and validates with Pydantic before returning. Invalid JSON raises
a 422 rather than silently passing bad data downstream.

**GCS persistence:** Every extraction is stored with a timestamp in GCS, enabling
historical comparison and audit trails. The `/extractions/{ticker}/latest`
endpoint lets downstream systems pull the most recent clean extraction.
