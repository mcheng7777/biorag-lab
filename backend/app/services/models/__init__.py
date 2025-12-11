"""
Model services for inference with fine-tuned bioinformatics code models.

Provides:
- Fine-tuned model loading and inference
- Model registry for version management
- Inference service with caching and batching
"""

from .fine_tuned_model import FineTunedModel, ModelInfo
from .model_registry import ModelRegistry, ModelVersion
from .inference_service import InferenceService, GenerationRequest, GenerationResponse, get_inference_service

__all__ = [
    "FineTunedModel",
    "ModelInfo",
    "ModelRegistry",
    "ModelVersion",
    "InferenceService",
    "GenerationRequest",
    "GenerationResponse",
    "get_inference_service",
]

