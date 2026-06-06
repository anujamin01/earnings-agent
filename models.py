from pydantic import BaseModel, field_validator
from typing import Any, Optional
from datetime import datetime


class ExtractedSignal(BaseModel):
    value: Any
    confidence: float
    reasoning: str
    quote: Optional[str] = None
    flagged: bool = False

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class EarningsSignals(BaseModel):
    ticker: str
    extracted_at: datetime = datetime.utcnow()
    signals: dict[str, ExtractedSignal]
    overall_confidence: float
    low_confidence_fields: list[str]
    raw_llm_output: str

    def is_trustworthy(self, threshold: float = 0.6) -> bool:
        """Returns False if overall confidence is below threshold -- do not feed to trading system."""
        return self.overall_confidence >= threshold

    def to_trading_payload(self) -> dict:
        """Clean payload safe to deliver downstream. Excludes low-confidence fields."""
        return {
            "ticker": self.ticker,
            "extracted_at": self.extracted_at.isoformat(),
            "overall_confidence": self.overall_confidence,
            "trustworthy": self.is_trustworthy(),
            "signals": {
                k: {
                    "value": v.value,
                    "confidence": v.confidence
                }
                for k, v in self.signals.items()
                if not v.flagged
            },
            "flagged_fields": self.low_confidence_fields
        }


class TranscriptRequest(BaseModel):
    ticker: str
    transcript: str
    source: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("transcript")
    @classmethod
    def transcript_min_length(cls, v: str) -> str:
        if len(v) < 200:
            raise ValueError("Transcript too short to extract meaningful signals")
        return v


class ExtractionResponse(BaseModel):
    ticker: str
    trustworthy: bool
    overall_confidence: float
    low_confidence_fields: list[str]
    signals: dict
    message: str
