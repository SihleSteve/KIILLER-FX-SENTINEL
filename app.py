import os, json, re, base64
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).parent
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

app = FastAPI(title="FX Sentinel AI Personal Scanner")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "ai_configured": client is not None, "model": MODEL}

@app.post("/api/analyze")
async def analyze_chart(image: UploadFile = File(...), timeframe: str = Form("M15"), symbol: str = Form("Unknown")):
    if client is None:
        raise HTTPException(status_code=503, detail="AI is not configured. Add OPENAI_API_KEY to your .env file and restart the server.")
    allowed = {"image/png", "image/jpeg", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPG or WEBP chart screenshot.")
    data = await image.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Please use an image under 12 MB.")
    mime = image.content_type
    b64 = base64.b64encode(data).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = f"""
You are FX Sentinel AI, a cautious chart-reading assistant. Analyze ONLY what is visible in the uploaded trading-chart screenshot.
User-selected symbol: {symbol}. User-selected timeframe: {timeframe}.
Do not pretend to know live prices, news, spread, broker conditions, or unseen candles. If axes/prices are unreadable, say so and do not invent exact price levels.
Give a directional technical interpretation, not a guarantee or personalized financial advice.
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
- Choose WAIT if the chart is unclear, cropped, conflicting, ranging without a clear edge, or missing enough context.
- Confidence is a qualitative evidence score, NOT a probability of profit. Never use 90+ unless the screenshot is exceptionally clear and aligned.
- Only provide numeric entry/SL/TP if readable prices and a defensible setup are visible. Otherwise use "Not reliably readable from screenshot".
- For BUY, explain bullish evidence; for SELL, bearish evidence; for WAIT, explain what confirmation is missing.
- Do not claim that a liquidity sweep, support/resistance, or pattern exists unless visible.
- Keep each explanation concise.
"""
    try:
        response = client.responses.create(
            model=MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url}
                ]
            }]
        )
        raw = response.output_text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        result = json.loads(raw)
        if result.get("direction") not in {"BUY", "SELL", "WAIT"}:
            raise ValueError("Invalid direction")
        result["confidence"] = max(0, min(100, int(result.get("confidence", 0))))
        result["timeframe_selected"] = timeframe
        result["symbol_selected"] = symbol
        result["disclaimer"] = "AI chart interpretation only. Verify on the live chart before any trade. No outcome is guaranteed."
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {str(e)[:240]}")
