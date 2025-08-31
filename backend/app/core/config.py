from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "BioRAG Lab"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = ["*"]  # TODO: Update for production
    
    # Supabase Configuration (to be used later)
    # SUPABASE_URL: str
    # SUPABASE_KEY: str
    
    # Gemini Configuration (to be used later)
    # GEMINI_API_KEY: str
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings"""
    return Settings()
