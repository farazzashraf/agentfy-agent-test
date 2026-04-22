from fastapi import FastAPI, HTTPException
from google.cloud import firestore

app = FastAPI()

# Initialize the Firestore client
# When this runs on Cloud Run, Google's internal network automatically authenticates it.
db = firestore.Client(project="agent-fy-494108")

@app.get("/")
async def root():
    return {"message": "Agent-fy Gateway is Live and listening."}

@app.get("/config/{tenant_id}")
async def get_tenant_config(tenant_id: str):
    """
    Fetches the specific business configuration (Memory) from Firestore.
    This is the critical step before routing any traffic to the AI.
    """
    doc_ref = db.collection("tenants").document(tenant_id)
    doc = doc_ref.get()

    if doc.exists:
        return {
            "status": "success",
            "tenant_id": tenant_id, 
            "memory": doc.to_dict()
        }
    else:
        raise HTTPException(status_code=404, detail="Tenant memory not found in the platform.")