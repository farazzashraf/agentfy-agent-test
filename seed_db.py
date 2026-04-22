from google.cloud import firestore

def seed_dummy_tenant():
    # Initialize the database client linked strictly to our project
    project_id = "agent-fy-494108"
    db = firestore.Client(project=project_id)

    # Our first dummy tenant
    tenant_id = "biz_889"
    
    # The exact configuration our Gateway will eventually inject into the AI container
    config_data = {
        "business_name": "The Grand Spice Restaurant",
        "subscription_tier": "Starter",
        "allowed_model": "gpt-4o-mini",
        "features": {
            "rag_engine": False,
            "memory": True
        },
        "secrets": {
            "gmail_connected": False 
        },
        "system_prompt": "You are a helpful booking assistant for The Grand Spice Restaurant. Keep your answers short, polite, and focused on booking tables."
    }

    print(f"Connecting to Firestore in project: {project_id}...")
    
    # Write the data to the 'tenants' collection
    db.collection("tenants").document(tenant_id).set(config_data)
    
    print(f"Success! Tenant '{tenant_id}' has been injected into the platform's memory.")

if __name__ == "__main__":
    seed_dummy_tenant()