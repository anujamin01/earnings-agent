import json
import logging
from datetime import datetime
from google.cloud import storage
from app.models import EarningsSignals

logger = logging.getLogger(__name__)


class GCSStore:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def save(self, result: EarningsSignals) -> str:
        timestamp = result.extracted_at.strftime("%Y%m%d_%H%M%S")
        blob_path = f"extractions/{result.ticker}/{timestamp}.json"

        payload = result.model_dump(mode="json")
        payload["extracted_at"] = result.extracted_at.isoformat()

        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json"
        )

        logger.info(f"Saved extraction for {result.ticker} to gs://{self.bucket.name}/{blob_path}")
        return f"gs://{self.bucket.name}/{blob_path}"

    def get_latest(self, ticker: str) -> dict | None:
        prefix = f"extractions/{ticker.upper()}/"
        blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))

        if not blobs:
            return None

        latest = sorted(blobs, key=lambda b: b.name)[-1]
        return json.loads(latest.download_as_text())

    def list_extractions(self, ticker: str) -> list[str]:
        prefix = f"extractions/{ticker.upper()}/"
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        return [b.name for b in blobs]
