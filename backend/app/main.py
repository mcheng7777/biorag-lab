from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return {
        "status": "healthy",
        "services": {
            "api": "ok",
            # We'll add more service checks here as we integrate them:
            # "supabase": "pending",
            # "gemini": "pending",
            # "faiss": "pending"
        }
    }
