# main.py — what developer writes
# No GCP, No Firestore, No project IDs. Nothing infrastructure-related.

import os
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Restaurant FAQ Agent")

# Only thing developer sets — their own Gemini key
# Agent-fy platform injects this automatically from their config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class ChatRequest(BaseModel):
    message: str
    injected_context: dict  # Agent-fy gateway always sends this


def build_system_prompt(config: dict) -> str:
    return f"""
You are a friendly FAQ assistant for {config.get('restaurant_name', 'our restaurant')}.

Answer customer questions accurately and warmly.
If you don't know something, say "I'm not sure — please call us at {config.get('phone_number', 'our number')}."

=== RESTAURANT DETAILS ===
Name: {config.get('restaurant_name')}
Cuisine: {config.get('cuisine_type')}
Address: {config.get('address')}
Phone: {config.get('phone_number')}
Price Range: {config.get('price_range')}

Opening Hours:
{config.get('opening_hours')}

Menu Highlights:
{config.get('menu_highlights')}

Dietary Options: {config.get('dietary_options', 'Please call us to ask.')}
Parking: {config.get('parking_info', 'Please contact us for parking info.')}
Reservation Policy: {config.get('reservation_policy', 'Please call us to book.')}

=== RULES ===
1. Be warm and concise. No long walls of text.
2. For prices, give the range and suggest calling for exact pricing.
3. For bookings, explain the policy and give the phone number.
4. If asked something unrelated, redirect: "I can only help with questions about {config.get('restaurant_name')}."
5. Never claim to be human. You are the restaurant's AI assistant.
""".strip()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/config")
async def get_config():
    try:
        with open("agentfy.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")


@app.get("/status")
async def status():
    return {"agent": "Restaurant FAQ Agent", "status": "live"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat/{tenant_id}")
async def chat(tenant_id: str, request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    config = request.injected_context
    system_prompt = build_system_prompt(config)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
        response = model.generate_content(request.message)

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "restaurant": config.get("restaurant_name"),
            "reply": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

# Mount static files (optional, if we add images/css files later)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
