import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
import google.generativeai as genai
from typing import Optional

app = FastAPI(title="Restaurant FAQ Agent")

# Platform injects GCP_PROJECT_ID — developer never sets this
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
db = firestore.Client(project=PROJECT_ID) if PROJECT_ID else None

# Developer's own Gemini key — injected by Agent-fy platform from their config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class ChatRequest(BaseModel):
    message: str
    # Agent-fy gateway injects tenant context in production
    # Falls back to Firestore lookup for local testing
    injected_context: Optional[dict] = None


def build_system_prompt(config: dict) -> str:
    """
    Builds a rich, specific system prompt from the business config.
    The better this prompt is, the better the agent answers.
    """
    return f"""
You are a friendly and knowledgeable FAQ assistant for {config.get('restaurant_name', 'our restaurant')}.

Your job is to answer customer questions accurately and warmly. 
Never make up information you don't have. If you don't know something, 
say "I'm not sure about that — please call us directly at {config.get('phone_number', 'our number')}."

=== RESTAURANT INFORMATION ===

Restaurant Name: {config.get('restaurant_name')}
Cuisine: {config.get('cuisine_type')}
Address: {config.get('address')}
Phone: {config.get('phone_number')}
Price Range: {config.get('price_range')}

Opening Hours:
{config.get('opening_hours')}

Our Menu Highlights:
{config.get('menu_highlights')}

Dietary Options We Offer:
{config.get('dietary_options', 'Please call us to ask about specific dietary requirements.')}

Parking:
{config.get('parking_info', 'Please contact us for parking information.')}

Reservation Policy:
{config.get('reservation_policy', 'Please call us to make a reservation.')}

=== YOUR BEHAVIOUR RULES ===

1. Be warm, helpful, and concise. Never give long walls of text.
2. If asked about menu prices, give the price range and suggest calling for exact prices.
3. If asked if you can book a table, explain the reservation policy and give the phone number.
4. If asked something completely unrelated to the restaurant, politely redirect: 
   "I can only help with questions about {config.get('restaurant_name')} — is there anything about us I can help with?"
5. Always end responses with a friendly touch when appropriate.
6. Never claim to be a human. If asked, say you're the restaurant's AI assistant.
""".strip()


@app.get("/")
async def root():
    return {
        "agent": "Restaurant FAQ Agent",
        "status": "live",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Agent-fy platform pings this to verify the agent is running."""
    return {"status": "healthy"}


@app.post("/chat/{tenant_id}")
async def chat(tenant_id: str, request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )

    # Production: Agent-fy gateway injects context
    # Local testing: read from Firestore directly
    if request.injected_context:
        config = request.injected_context
    elif db:
        doc = db.collection("tenants").document(tenant_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
        config = doc.to_dict()
    else:
        raise HTTPException(
            status_code=500,
            detail="No config source available. Set GCP_PROJECT_ID or pass injected_context."
        )

    system_prompt = build_system_prompt(config)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
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
        raise HTTPException(
            status_code=500,
            detail=f"AI generation failed: {str(e)}"
        )