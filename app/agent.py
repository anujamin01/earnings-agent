import json
import re
import anthropic
from app.models import EarningsSignals, ExtractedSignal


SYSTEM_PROMPT = """You are a financial data extraction agent. Your job is to extract
clean, structured trading signals from earnings call transcripts.

You must follow the ReAct pattern:
- Thought: reason about what to extract next
- Action: extract a specific piece of information
- Observation: what you found
- Repeat until all signals are extracted

Always be precise. If a value is ambiguous or not mentioned, say so explicitly.
Never hallucinate numbers. If you are not confident, lower your confidence score.

Respond ONLY in valid JSON matching the schema provided."""


EXTRACTION_PROMPT = """Extract trading signals from the following earnings call transcript.

For each signal, provide:
- value: the extracted value (string, number, or null if not found)
- confidence: float 0.0-1.0 (1.0 = explicitly stated, 0.5 = implied, 0.0 = not found)
- reasoning: brief explanation of why you extracted this value
- quote: the exact quote from the transcript supporting this (null if not found)

Signals to extract:
1. revenue_guidance: Forward revenue guidance (next quarter or year)
2. eps_guidance: Forward EPS guidance
3. revenue_actual: Actual revenue reported this quarter
4. eps_actual: Actual EPS reported this quarter
5. management_tone: Overall tone (bullish/neutral/bearish) with reasoning
6. key_risks: Top 3 risks mentioned by management
7. key_opportunities: Top 3 opportunities mentioned by management
8. guidance_raised: Boolean - did they raise guidance? (true/false/null if no prior guidance)
9. buyback_announced: Boolean - was a share buyback announced or mentioned?
10. dividend_change: Any dividend change mentioned (increase/decrease/none)

Return ONLY a JSON object with these exact keys. No markdown, no explanation outside JSON.

Transcript:
{transcript}"""


class EarningsExtractionAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5"

    def extract(self, transcript: str, ticker: str) -> EarningsSignals:
        prompt = EXTRACTION_PROMPT.format(transcript=transcript[:12000])

        raw_response = self._call_llm(prompt)
        parsed = self._parse_and_validate(raw_response, ticker)
        return parsed

    def _call_llm(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _parse_and_validate(self, raw: str, ticker: str) -> EarningsSignals:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("```").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")

        signals = {}
        for key, val in data.items():
            if isinstance(val, dict):
                confidence = float(val.get("confidence", 0.0))
                signals[key] = ExtractedSignal(
                    value=val.get("value"),
                    confidence=confidence,
                    reasoning=val.get("reasoning", ""),
                    quote=val.get("quote"),
                    flagged=confidence < 0.5
                )

        overall_confidence = (
            sum(s.confidence for s in signals.values()) / len(signals)
            if signals else 0.0
        )

        return EarningsSignals(
            ticker=ticker,
            signals=signals,
            overall_confidence=round(overall_confidence, 3),
            low_confidence_fields=[k for k, v in signals.items() if v.flagged],
            raw_llm_output=raw
        )
