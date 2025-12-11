"""
Model registry for managing fine-tuned model versions.

Provides:
- Model version tracking
- Model metadata storage
- Active model selection
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a version of a fine-tuned model."""

    version_id: str
    name: str
    base_model: str
    adapter_path: str
    created_at: datetime
    description: str = ""
    metrics: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_active: bool = False

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "metrics": self.metrics,
            "tags": self.tags,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelRegistry:
    """
    Registry for managing fine-tuned model versions.
    """

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self._versions: dict[str, ModelVersion] = {}
        self._active_version: Optional[str] = None
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                for version_data in data.get("versions", []):
                    version = ModelVersion.from_dict(version_data)
                    self._versions[version.version_id] = version
                    if version.is_active:
                        self._active_version = version.version_id

            logger.info(f"Loaded registry with {len(self._versions)} versions")

    def _save_registry(self):
        """Save registry to disk."""
        data = {
            "versions": [v.to_dict() for v in self._versions.values()],
            "active_version": self._active_version,
        }
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        base_model: str,
        adapter_path: str,
        description: str = "",
        metrics: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        set_active: bool = False,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Display name for the model
            base_model: Base model identifier
            adapter_path: Path to LoRA adapters
            description: Model description
            metrics: Training/evaluation metrics
            tags: Model tags
            set_active: Whether to set as active model

        Returns:
            Created ModelVersion
        """
        # Use microseconds for unique version IDs
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        version = ModelVersion(
            version_id=version_id,
            name=name,
            base_model=base_model,
            adapter_path=adapter_path,
            created_at=datetime.now(),
            description=description,
            metrics=metrics or {},
            tags=tags or [],
            is_active=set_active,
        )

        self._versions[version_id] = version

        if set_active:
            self._set_active(version_id)

        self._save_registry()
        logger.info(f"Registered model version: {version_id}")

        return version

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(version_id)

    def get_active(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        if self._active_version:
            return self._versions.get(self._active_version)
        return None

    def set_active(self, version_id: str) -> bool:
        """Set a model version as active."""
        if version_id not in self._versions:
            logger.error(f"Version not found: {version_id}")
            return False

        self._set_active(version_id)
        self._save_registry()
        logger.info(f"Set active version: {version_id}")
        return True

    def _set_active(self, version_id: str):
        """Internal method to set active version."""
        # Deactivate current
        if self._active_version and self._active_version in self._versions:
            self._versions[self._active_version].is_active = False

        # Activate new
        self._active_version = version_id
        self._versions[version_id].is_active = True

    def list_versions(
        self,
        tags: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[ModelVersion]:
        """
        List model versions.

        Args:
            tags: Filter by tags
            limit: Maximum versions to return

        Returns:
            List of ModelVersion objects
        """
        versions = list(self._versions.values())

        # Filter by tags
        if tags:
            versions = [v for v in versions if any(t in v.tags for t in tags)]

        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions[:limit]

    def delete_version(self, version_id: str) -> bool:
        """Delete a model version."""
        if version_id not in self._versions:
            return False

        version = self._versions[version_id]

        # Don't delete active version
        if version.is_active:
            logger.error("Cannot delete active version")
            return False

        del self._versions[version_id]
        self._save_registry()
        logger.info(f"Deleted version: {version_id}")

        return True

    def update_metrics(
        self,
        version_id: str,
        metrics: dict,
    ) -> bool:
        """Update metrics for a model version."""
        if version_id not in self._versions:
            return False

        self._versions[version_id].metrics.update(metrics)
        self._save_registry()
        return True

    def get_best_version(self, metric: str = "avg_score") -> Optional[ModelVersion]:
        """Get the version with the best metric value."""
        best_version = None
        best_value = -1

        for version in self._versions.values():
            value = version.metrics.get(metric, 0)
            if value > best_value:
                best_value = value
                best_version = version

        return best_version

    def compare_versions(
        self,
        version_ids: list[str],
    ) -> dict:
        """Compare metrics across versions."""
        comparison = {}

        for version_id in version_ids:
            version = self._versions.get(version_id)
            if version:
                comparison[version_id] = {
                    "name": version.name,
                    "created_at": version.created_at.isoformat(),
                    "metrics": version.metrics,
                }

        return comparison

