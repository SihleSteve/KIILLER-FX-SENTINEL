
import os
import json
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

ROOT = Path(__file__).parent
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

app = FastAPI(title="FX Sentinel AI Personal Scanner")

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static"
)


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ai_configured": client is not None,
        "model": MODEL
    }


@app.post("/api/analyze")
async def analyze_chart(
    image: UploadFile = File(...),
    timeframe: str = Form("M15"),
    symbol: str = Form("Unknown")
):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI is not configured. Add GEMINI_API_KEY in Render."
        )

    allowed = {"image/png", "image/jpeg", "image/webp"}

    if image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPG or WEBP chart screenshot."
        )

    data = await image.read()

    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Please use an image under 12 MB."
        )

    mime = image.content_type

    prompt = f"""
You are FX Sentinel AI, a cautious chart-reading assistant.
Analyze ONLY what is visible in the uploaded trading-chart screenshot.

User-selected symbol: {symbol}.
User-selected timeframe: {timeframe}.

Do not pretend to know live prices, news, spread, broker
conditions, or unseen candles.

If axes or prices are unreadable, do not invent exact price levels.

Give a directional technical interpretation, not a guarantee
or personalized financial advice.

Return ONLY valid JSON with these keys:

{{
 "direction": "BUY" | "SELL" | "WAIT",
 "confidence": integer from 0 to 100,
 "symbol_detected": string,
 "timeframe_detected": string,
 "trend": string,
 "structure": string,
 "liquidity": string,
 "key_levels": string,
 "entry_zone": string,
 "stop_loss": string,
 "take_profit": string,
 "reasoning": [string, string, string],
 "invalid_if": string,
 "limitations": [string]
}}

Rules:
- Choose WAIT if the chart is unclear, cropped, conflicting,
  ranging without a clear edge, or missing enough context.
- Confidence is an evidence score, NOT a probability of profit.
- Never use 90+ unless the screenshot is exceptionally clear
  and aligned.
- Only provide numeric entry/SL/TP if readable prices and a
  defensible setup are visible.
- Otherwise use "Not reliably readable from screenshot".
- For BUY, explain bullish evidence.
- For SELL, explain bearish evidence.
- For WAIT, explain what confirmation is missing.
- Do not claim a liquidity sweep, support/resistance, or
  pattern exists unless visible.
- Keep each explanation concise.
"""

    try:
        image_part = types.Part.from_bytes(
            data=data,
            mime_type=mime
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )

        raw = (response.text or "").strip()

        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.I
        )

        result = json.loads(raw)

        if result.get("direction") not in {"BUY", "SELL", "WAIT"}:
            raise ValueError("Invalid direction returned by Gemini")

        result["confidence"] = max(
            0,
            min(100, int(result.get("confidence", 0)))
        )

        result["timeframe_selected"] = timeframe
        result["symbol_selected"] = symbol

        result["disclaimer"] = (
            "AI chart interpretation only. Verify on the live chart "
            "before any trade. No outcome is guaranteed."
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini analysis failed: {str(e)[:240]}"
        )
