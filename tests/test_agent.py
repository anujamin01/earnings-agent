import pytest
import json
from unittest.mock import MagicMock, patch
from app.agent import EarningsExtractionAgent
from app.models import EarningsSignals, ExtractedSignal, TranscriptRequest


SAMPLE_TRANSCRIPT = """
Good morning everyone and thank you for joining Apple's Q3 2025 earnings call.
I'm Tim Cook, and with me today is our CFO Luca Maestri.

We're thrilled to report record revenue of $94.8 billion for the quarter,
up 8% year over year. Our EPS came in at $1.53, beating estimates of $1.39.

Looking ahead to Q4, we expect revenue in the range of $89 to $91 billion,
representing growth of 4 to 5% year over year. We're also raising our full
year EPS guidance to $6.10 from our prior guidance of $5.95.

Our board has approved an additional $90 billion share repurchase program.
The dividend remains unchanged at $0.25 per share.

Key risks we're watching include macroeconomic uncertainty in China, foreign
exchange headwinds, and supply chain constraints for our Vision Pro components.

We're excited about opportunities in generative AI integration across our
product line, continued Services growth which hit $24.2 billion this quarter,
and expansion in emerging markets particularly India where we saw 35% growth.

Management remains very confident in our trajectory. The iPhone 16 cycle has
exceeded our expectations and we're entering Q4 with strong momentum.
""" * 2


MOCK_LLM_RESPONSE = json.dumps({
    "revenue_actual": {
        "value": "$94.8 billion",
        "confidence": 1.0,
        "reasoning": "Explicitly stated as record revenue for the quarter",
        "quote": "record revenue of $94.8 billion for the quarter"
    },
    "eps_actual": {
        "value": "$1.53",
        "confidence": 1.0,
        "reasoning": "Explicitly stated EPS figure",
        "quote": "Our EPS came in at $1.53"
    },
    "revenue_guidance": {
        "value": "$89-91 billion",
        "confidence": 1.0,
        "reasoning": "Explicit Q4 guidance range provided",
        "quote": "we expect revenue in the range of $89 to $91 billion"
    },
    "eps_guidance": {
        "value": "$6.10 full year",
        "confidence": 0.95,
        "reasoning": "Full year EPS guidance raised explicitly",
        "quote": "raising our full year EPS guidance to $6.10"
    },
    "management_tone": {
        "value": "bullish",
        "confidence": 0.9,
        "reasoning": "Strong language around momentum, confidence, and exceeding expectations",
        "quote": "Management remains very confident in our trajectory"
    },
    "key_risks": {
        "value": ["China macroeconomic uncertainty", "Foreign exchange headwinds", "Vision Pro supply chain constraints"],
        "confidence": 1.0,
        "reasoning": "Three risks explicitly listed by management",
        "quote": "Key risks we're watching include macroeconomic uncertainty in China"
    },
    "key_opportunities": {
        "value": ["Generative AI integration", "Services growth", "India expansion"],
        "confidence": 1.0,
        "reasoning": "Three opportunities explicitly stated",
        "quote": "We're excited about opportunities in generative AI integration"
    },
    "guidance_raised": {
        "value": True,
        "confidence": 1.0,
        "reasoning": "EPS guidance explicitly raised from $5.95 to $6.10",
        "quote": "raising our full year EPS guidance to $6.10 from our prior guidance of $5.95"
    },
    "buyback_announced": {
        "value": True,
        "confidence": 1.0,
        "reasoning": "$90 billion buyback program announced",
        "quote": "board has approved an additional $90 billion share repurchase program"
    },
    "dividend_change": {
        "value": "none",
        "confidence": 1.0,
        "reasoning": "Dividend explicitly stated as unchanged",
        "quote": "The dividend remains unchanged at $0.25 per share"
    }
})


class TestEarningsExtractionAgent:
    def setup_method(self):
        self.agent = EarningsExtractionAgent(api_key="test-key")

    def test_parse_valid_llm_response(self):
        result = self.agent._parse_and_validate(MOCK_LLM_RESPONSE, "AAPL")
        assert result.ticker == "AAPL"
        assert "revenue_actual" in result.signals
        assert result.signals["revenue_actual"].value == "$94.8 billion"
        assert result.signals["revenue_actual"].confidence == 1.0
        assert not result.signals["revenue_actual"].flagged

    def test_overall_confidence_calculated(self):
        result = self.agent._parse_and_validate(MOCK_LLM_RESPONSE, "AAPL")
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_low_confidence_fields_flagged(self):
        low_conf_response = json.loads(MOCK_LLM_RESPONSE)
        low_conf_response["revenue_guidance"]["confidence"] = 0.3
        result = self.agent._parse_and_validate(json.dumps(low_conf_response), "AAPL")
        assert "revenue_guidance" in result.low_confidence_fields
        assert result.signals["revenue_guidance"].flagged

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            self.agent._parse_and_validate("this is not json", "AAPL")

    def test_markdown_json_stripped(self):
        wrapped = f"```json\n{MOCK_LLM_RESPONSE}\n```"
        result = self.agent._parse_and_validate(wrapped, "AAPL")
        assert result.ticker == "AAPL"

    @patch("app.agent.anthropic.Anthropic")
    def test_extract_calls_llm(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value.content = [
            MagicMock(text=MOCK_LLM_RESPONSE)
        ]
        agent = EarningsExtractionAgent(api_key="test-key")
        result = agent.extract(transcript=SAMPLE_TRANSCRIPT, ticker="AAPL")
        assert result.ticker == "AAPL"
        mock_client.messages.create.assert_called_once()


class TestEarningsSignals:
    def _make_signal(self, confidence: float) -> ExtractedSignal:
        return ExtractedSignal(
            value="test",
            confidence=confidence,
            reasoning="test",
            flagged=confidence < 0.5
        )

    def test_is_trustworthy_above_threshold(self):
        signals = {k: self._make_signal(0.9) for k in ["a", "b", "c"]}
        result = EarningsSignals(
            ticker="AAPL",
            signals=signals,
            overall_confidence=0.9,
            low_confidence_fields=[],
            raw_llm_output=""
        )
        assert result.is_trustworthy()

    def test_is_not_trustworthy_below_threshold(self):
        signals = {k: self._make_signal(0.3) for k in ["a", "b", "c"]}
        result = EarningsSignals(
            ticker="AAPL",
            signals=signals,
            overall_confidence=0.3,
            low_confidence_fields=["a", "b", "c"],
            raw_llm_output=""
        )
        assert not result.is_trustworthy()

    def test_trading_payload_excludes_flagged(self):
        signals = {
            "revenue": self._make_signal(0.9),
            "guidance": self._make_signal(0.2)
        }
        signals["guidance"].flagged = True
        result = EarningsSignals(
            ticker="AAPL",
            signals=signals,
            overall_confidence=0.55,
            low_confidence_fields=["guidance"],
            raw_llm_output=""
        )
        payload = result.to_trading_payload()
        assert "revenue" in payload["signals"]
        assert "guidance" not in payload["signals"]
        assert "guidance" in payload["flagged_fields"]


class TestTranscriptRequest:
    def test_ticker_uppercased(self):
        req = TranscriptRequest(ticker="aapl", transcript="x" * 300)
        assert req.ticker == "AAPL"

    def test_short_transcript_rejected(self):
        with pytest.raises(Exception):
            TranscriptRequest(ticker="AAPL", transcript="too short")

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            ExtractedSignal(value="x", confidence=1.5, reasoning="test")
