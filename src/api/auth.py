from fastapi import Header, HTTPException, Depends
import os

# Get token from env var
API_TOKEN = os.getenv("AMGIX_AUTH_TOKEN")

async def verify_token(authorization: str = Header(None)):
    # If no token is set in ENV, we skip check (Dev mode)
    if not API_TOKEN:
        return
    
    # Check for "Bearer <token>" or just the raw token
    if authorization != f"Bearer {API_TOKEN}" and authorization != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Amgix Token")

# Then in your main app:
# app.include_router(query_router, dependencies=[Depends(verify_token)])