from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class ReasoningLevel(str, Enum):
    """Reasoning levels supported by GPT-OSS"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelMode(str, Enum):
    """Model operation modes"""
    DOCUMENTATION = "documentation"
    PAPER = "paper"

class ModelConfig(BaseModel):
    """Model-specific configuration"""
    model_id: str = "openai/gpt-oss-20b"
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    do_sample: bool = True
    use_cache: bool = True

class InferenceConfig(BaseModel):
    """Inference-specific configuration"""
    timeout_seconds: int = Field(default=30, ge=1)
    max_batch_size: int = Field(default=1, ge=1)
    max_concurrent_requests: int = Field(default=5, ge=1)
    device: str = "cuda"  # Will fall back to CPU if CUDA not available
    quantization: Optional[str] = "int8"  # Optional quantization for memory efficiency

class Settings(BaseSettings):
    """Global settings for model service"""
    # Model settings
    model_config: ModelConfig = ModelConfig()
    inference_config: InferenceConfig = InferenceConfig()
    
    # Resource limits
    max_memory_gb: float = Field(default=16.0, ge=1.0)
    max_gpu_memory_gb: float = Field(default=16.0, ge=1.0)
    
    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    metrics_port: int = 9090
    
    # Cache settings
    cache_dir: str = ".model_cache"
    max_cache_size_gb: float = Field(default=10.0, ge=0.0)
    
    class Config:
        env_prefix = "MODEL_"
        env_file = ".env"

# Create settings instance
settings = Settings()
