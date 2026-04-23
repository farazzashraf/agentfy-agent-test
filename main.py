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
    history: list = []  # Added to match Playground payload
    injected_context: dict = {}  # Made optional for Playground compatibility


def build_system_prompt(config: dict) -> str:
    # Use fallback if config is empty (Playground case)
    restaurant_name = config.get('restaurant_name', 'our restaurant')
    return f"""
You are a friendly FAQ assistant for {restaurant_name}.

Answer customer questions accurately and warmly.
If you don't know something, say "I'm not sure — please call us at {config.get('phone_number', 'our number')}."

=== RESTAURANT DETAILS ===
Name: {restaurant_name}
Cuisine: {config.get('cuisine_type', 'N/A')}
Address: {config.get('address', 'N/A')}
Phone: {config.get('phone_number', 'N/A')}
Price Range: {config.get('price_range', 'N/A')}

Opening Hours:
{config.get('opening_hours', 'N/A')}

Menu Highlights:
{config.get('menu_highlights', 'N/A')}

Dietary Options: {config.get('dietary_options', 'Please call us to ask.')}
Parking: {config.get('parking_info', 'Please contact us for parking info.')}
Reservation Policy: {config.get('reservation_policy', 'Please call us to book.')}

=== RULES ===
1. Be warm and concise. No long walls of text.
2. For prices, give the range and suggest calling for exact pricing.
3. For bookings, explain the policy and give the phone number.
4. If asked something unrelated, redirect: "I can only help with questions about {restaurant_name}."
5. Never claim to be human. You are the restaurant's AI assistant.
""".strip()


@app.get("/")
async def root_get():
    return FileResponse("static/index.html")


@app.post("/")
async def root_post(request: ChatRequest):
    """
    Standardized chat endpoint. 
    Handles POST at root to match the Playground's default target_url.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    # Configure proxy if BASE_URL is provided
    base_url = os.getenv("GEMINI_BASE_URL")
    if base_url:
        from google.api_core import client_options
        # Ensure we use REST transport for proxying
        genai.configure(
            api_key=GEMINI_API_KEY,
            transport='rest',
            client_options=client_options.ClientOptions(api_endpoint=base_url)
        )

    config = request.injected_context
    system_prompt = build_system_prompt(config)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", # Fixed model name to 1.5
            system_instruction=system_prompt
        )
        
        # Include history if available (Gemini SDK format: [{'role': 'user', 'parts': [...]}, ...])
        # For simplicity, we just send the single message for now or implement chat session
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(request.message)

        return {
            "status": "success",
            "response": response.text
        }

    except Exception as e:
        # Log error for debugging (visible in Cloud Run logs)
        print(f"Error calling Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


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


# Mount static files (optional, if we add images/css files later)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
