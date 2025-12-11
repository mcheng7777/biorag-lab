from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1 import training

app = FastAPI(
    title="BioRAG Lab API",
    description="API for BioRAG Lab - A reinforcement-learning powered RAG application for bioinformatics",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Update this with actual frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(training.router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "status": "ok",
        "message": "BioRAG Lab API is running",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    inference_status = "unknown"
    try:
        from .services.models import get_inference_service
        service = get_inference_service()
        inference_status = "ready" if service.is_ready() else "no_model_loaded"
    except Exception as e:
        inference_status = f"error: {str(e)[:50]}"

    return {
        "status": "healthy",
        "services": {
            "api": "ok",
            "inference": inference_status,
            # We'll add more service checks here as we integrate them:
            # "supabase": "pending",
            # "gemini": "pending",
            # "faiss": "pending"
        }
    }
