import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
import google.generativeai as genai

app = FastAPI()

# 1. Initialize the Platform Memory
db = firestore.Client(project="agent-fy-494108")

# 2. Configure the AI Engine securely
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 3. Define the incoming message structure
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Agent-fy Gateway is Live and listening."}

@app.get("/config/{tenant_id}")
async def get_tenant_config(tenant_id: str):
    doc_ref = db.collection("tenants").document(tenant_id)
    doc = doc_ref.get()
    if doc.exists:
        return {"status": "success", "tenant_id": tenant_id, "memory": doc.to_dict()}
    raise HTTPException(status_code=404, detail="Tenant memory not found in the platform.")

@app.post("/chat/{tenant_id}")
async def chat_proxy(tenant_id: str, request: ChatRequest):
    """
    The Agent-fy LLM Proxy Core.
    Routes user traffic to Gemini using the business's specific rules.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Agent-fy AI Proxy is missing its API Key.")

    # A. Fetch the Business Rules (Memory)
    doc_ref = db.collection("tenants").document(tenant_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Tenant memory not found.")
        
    tenant_data = doc.to_dict()
    system_prompt = tenant_data.get("system_prompt", "You are a helpful AI assistant.")
    
    try:
        # B. Boot up the Gemini Model with the Business's rules
        # Using gemini-1.5-flash as it is lightning fast and free-tier eligible
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
        
        # C. Forward the user's prompt to the AI
        response = model.generate_content(request.message)
        
        # D. Return the sanitized response to the frontend
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "business_name": tenant_data.get("business_name"),
            "reply": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy Generation Error: {str(e)}")