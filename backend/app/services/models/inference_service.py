"""
Inference service for bioinformatics code generation.

Provides:
- Code generation API
- Model switching
- Request handling
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .fine_tuned_model import FineTunedModel, ModelInfo
from .model_registry import ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Request for code generation."""

    instruction: str
    context: str = ""
    language: str = "python"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class GenerationResponse:
    """Response from code generation."""

    code: str
    language: str
    model_id: str
    generated_at: datetime
    generation_time_ms: float
    tokens_generated: int

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "language": self.language,
            "model_id": self.model_id,
            "generated_at": self.generated_at.isoformat(),
            "generation_time_ms": self.generation_time_ms,
            "tokens_generated": self.tokens_generated,
        }


class InferenceService:
    """
    Service for code generation inference.

    Manages model loading and provides generation API.
    """

    def __init__(
        self,
        registry_dir: str = "models/registry",
        cache_dir: str = ".cache/huggingface",
        auto_load: bool = True,
    ):
        self.registry = ModelRegistry(registry_dir)
        self.model_service = FineTunedModel(cache_dir=cache_dir)
        self._current_version: Optional[ModelVersion] = None

        if auto_load:
            self._load_active_model()

    def _load_active_model(self):
        """Load the active model from registry."""
        active = self.registry.get_active()
        if active:
            self.load_version(active.version_id)

    def load_version(self, version_id: str) -> bool:
        """
        Load a specific model version.

        Args:
            version_id: Version ID to load

        Returns:
            True if successful
        """
        version = self.registry.get_version(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False

        try:
            self.model_service.load_fine_tuned(
                base_model=version.base_model,
                adapter_path=version.adapter_path,
            )
            self._current_version = version
            logger.info(f"Loaded model version: {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_base_model(self, model_name: str) -> bool:
        """
        Load a base model without fine-tuning.

        Args:
            model_name: Hugging Face model name

        Returns:
            True if successful
        """
        try:
            self.model_service.load_base_model(model_name)
            self._current_version = None
            logger.info(f"Loaded base model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate code for a request.

        Args:
            request: GenerationRequest object

        Returns:
            GenerationResponse with generated code
        """
        import time

        if not self.model_service.is_loaded():
            raise RuntimeError("No model loaded")

        start_time = time.time()

        # Add language hint to instruction if needed
        instruction = request.instruction
        if request.language and request.language.lower() not in instruction.lower():
            instruction = f"[{request.language.upper()}] {instruction}"

        # Generate
        code = self.model_service.generate(
            instruction=instruction,
            context=request.context,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        generation_time = (time.time() - start_time) * 1000

        # Estimate tokens (rough approximation)
        tokens_generated = len(code.split()) * 1.3

        return GenerationResponse(
            code=code,
            language=request.language,
            model_id=self.get_current_model_id(),
            generated_at=datetime.now(),
            generation_time_ms=generation_time,
            tokens_generated=int(tokens_generated),
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """
        Async wrapper for generation.

        Args:
            request: GenerationRequest object

        Returns:
            GenerationResponse with generated code
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)

    def get_current_model_id(self) -> str:
        """Get ID of the currently loaded model."""
        if self._current_version:
            return self._current_version.version_id

        info = self.model_service.get_info()
        if info:
            return info.model_id

        return "unknown"

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get the currently loaded model version."""
        return self._current_version

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get info about the currently loaded model."""
        return self.model_service.get_info()

    def list_available_models(self) -> list[dict]:
        """List all available model versions."""
        versions = self.registry.list_versions(limit=100)
        return [
            {
                "version_id": v.version_id,
                "name": v.name,
                "base_model": v.base_model,
                "is_active": v.is_active,
                "created_at": v.created_at.isoformat(),
                "metrics": v.metrics,
            }
            for v in versions
        ]

    def switch_model(self, version_id: str) -> bool:
        """
        Switch to a different model version.

        Args:
            version_id: Version ID to switch to

        Returns:
            True if successful
        """
        # Unload current model
        self.model_service.unload()

        # Load new version
        success = self.load_version(version_id)

        if success:
            # Update active in registry
            self.registry.set_active(version_id)

        return success

    def unload_model(self):
        """Unload the current model."""
        self.model_service.unload()
        self._current_version = None

    def is_ready(self) -> bool:
        """Check if the service is ready for inference."""
        return self.model_service.is_loaded()

    def get_status(self) -> dict:
        """Get service status."""
        info = self.model_service.get_info()

        return {
            "ready": self.is_ready(),
            "current_model": self.get_current_model_id() if self.is_ready() else None,
            "model_info": info.to_dict() if info else None,
            "available_versions": len(self.registry.list_versions(limit=100)),
        }


# Singleton instance for the application
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get the inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(auto_load=False)
    return _inference_service

