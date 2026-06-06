import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.agent import EarningsExtractionAgent
from app.models import TranscriptRequest, ExtractionResponse
from app.storage import GCSStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent: EarningsExtractionAgent = None
store: GCSStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, store
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    agent = EarningsExtractionAgent(api_key=api_key)

    bucket = os.environ.get("GCS_BUCKET_NAME")
    if bucket:
        store = GCSStore(bucket_name=bucket)
        logger.info(f"GCS store initialized: {bucket}")
    else:
        logger.warning("GCS_BUCKET_NAME not set -- results will not be persisted")

    yield


app = FastAPI(
    title="Earnings Extraction Agent",
    description="Extracts structured trading signals from earnings call transcripts using LLMs",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResponse)
def extract(request: TranscriptRequest):
    logger.info(f"Extraction request for ticker: {request.ticker}")

    try:
        result = agent.extract(
            transcript=request.transcript,
            ticker=request.ticker
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Extraction failed for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail="Extraction failed")

    gcs_path = None
    if store:
        try:
            gcs_path = store.save(result)
        except Exception as e:
            logger.warning(f"Failed to persist to GCS: {e}")

    trading_payload = result.to_trading_payload()

    message = (
        f"Extraction complete. Overall confidence: {result.overall_confidence:.0%}. "
        + (f"Stored at {gcs_path}." if gcs_path else "GCS storage skipped.")
        + (f" WARNING: {len(result.low_confidence_fields)} low-confidence fields flagged." if result.low_confidence_fields else "")
    )

    return ExtractionResponse(
        ticker=result.ticker,
        trustworthy=result.is_trustworthy(),
        overall_confidence=result.overall_confidence,
        low_confidence_fields=result.low_confidence_fields,
        signals=trading_payload["signals"],
        message=message
    )


@app.get("/extractions/{ticker}/latest")
def get_latest(ticker: str):
    if not store:
        raise HTTPException(status_code=503, detail="GCS storage not configured")
    result = store.get_latest(ticker)
    if not result:
        raise HTTPException(status_code=404, detail=f"No extractions found for {ticker}")
    return JSONResponse(content=result)


@app.get("/extractions/{ticker}")
def list_extractions(ticker: str):
    if not store:
        raise HTTPException(status_code=503, detail="GCS storage not configured")
    files = store.list_extractions(ticker)
    return {"ticker": ticker.upper(), "extractions": files}
