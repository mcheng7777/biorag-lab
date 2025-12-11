"""Tests for the ModelRegistry class."""

import pytest
import tempfile
import shutil
from pathlib import Path

from app.services.models.model_registry import ModelRegistry, ModelVersion


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for registry."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def registry(self, temp_dir):
        """Create a ModelRegistry instance."""
        return ModelRegistry(registry_dir=temp_dir)

    # Registration Tests
    def test_register_model(self, registry):
        """Test registering a new model."""
        version = registry.register(
            name="test-model",
            base_model="codellama/CodeLlama-7b-hf",
            adapter_path="/path/to/adapter",
            description="Test model for unit tests",
        )

        assert version is not None
        assert version.name == "test-model"
        assert version.base_model == "codellama/CodeLlama-7b-hf"
        assert version.is_active is False

    def test_register_with_metrics(self, registry):
        """Test registering model with metrics."""
        metrics = {"eval_loss": 0.5, "accuracy": 0.85}

        version = registry.register(
            name="model-with-metrics",
            base_model="codellama/CodeLlama-7b-hf",
            adapter_path="/path/to/adapter",
            metrics=metrics,
        )

        assert version.metrics["eval_loss"] == 0.5
        assert version.metrics["accuracy"] == 0.85

    def test_register_with_tags(self, registry):
        """Test registering model with tags."""
        version = registry.register(
            name="tagged-model",
            base_model="codellama/CodeLlama-7b-hf",
            adapter_path="/path/to/adapter",
            tags=["bioinformatics", "python", "v1"],
        )

        assert "bioinformatics" in version.tags
        assert len(version.tags) == 3

    def test_register_and_set_active(self, registry):
        """Test registering and immediately setting as active."""
        version = registry.register(
            name="active-model",
            base_model="codellama/CodeLlama-7b-hf",
            adapter_path="/path/to/adapter",
            set_active=True,
        )

        assert version.is_active is True
        assert registry.get_active() == version

    # Version Retrieval Tests
    def test_get_version(self, registry):
        """Test getting a specific version."""
        registered = registry.register(
            name="test",
            base_model="base",
            adapter_path="/path",
        )

        retrieved = registry.get_version(registered.version_id)

        assert retrieved is not None
        assert retrieved.version_id == registered.version_id

    def test_get_nonexistent_version(self, registry):
        """Test getting a version that doesn't exist."""
        result = registry.get_version("nonexistent_id")

        assert result is None

    def test_list_versions(self, registry):
        """Test listing all versions."""
        for i in range(5):
            registry.register(
                name=f"model-{i}",
                base_model="base",
                adapter_path=f"/path/{i}",
            )

        versions = registry.list_versions(limit=10)

        assert len(versions) == 5

    def test_list_versions_with_limit(self, registry):
        """Test listing versions with limit."""
        for i in range(10):
            registry.register(
                name=f"model-{i}",
                base_model="base",
                adapter_path=f"/path/{i}",
            )

        versions = registry.list_versions(limit=3)

        assert len(versions) == 3

    def test_list_versions_by_tag(self, registry):
        """Test filtering versions by tag."""
        registry.register(
            name="python-model",
            base_model="base",
            adapter_path="/path/1",
            tags=["python"],
        )
        registry.register(
            name="r-model",
            base_model="base",
            adapter_path="/path/2",
            tags=["r"],
        )

        python_versions = registry.list_versions(tags=["python"])

        assert len(python_versions) == 1
        assert "python" in python_versions[0].tags

    # Active Model Tests
    def test_set_active(self, registry):
        """Test setting active model."""
        v1 = registry.register(name="v1", base_model="base", adapter_path="/p1")
        v2 = registry.register(name="v2", base_model="base", adapter_path="/p2")

        registry.set_active(v1.version_id)
        assert registry.get_active().version_id == v1.version_id

        registry.set_active(v2.version_id)
        assert registry.get_active().version_id == v2.version_id

        # v1 should no longer be active
        v1_updated = registry.get_version(v1.version_id)
        assert v1_updated.is_active is False

    def test_set_active_nonexistent(self, registry):
        """Test setting nonexistent version as active."""
        result = registry.set_active("nonexistent_id")

        assert result is False

    # Delete Tests
    def test_delete_version(self, registry):
        """Test deleting a version."""
        version = registry.register(
            name="to-delete",
            base_model="base",
            adapter_path="/path",
        )

        result = registry.delete_version(version.version_id)

        assert result is True
        assert registry.get_version(version.version_id) is None

    def test_delete_active_version_fails(self, registry):
        """Test that deleting active version fails."""
        version = registry.register(
            name="active",
            base_model="base",
            adapter_path="/path",
            set_active=True,
        )

        result = registry.delete_version(version.version_id)

        assert result is False
        assert registry.get_version(version.version_id) is not None

    def test_delete_nonexistent(self, registry):
        """Test deleting nonexistent version."""
        result = registry.delete_version("nonexistent_id")

        assert result is False

    # Metrics Update Tests
    def test_update_metrics(self, registry):
        """Test updating model metrics."""
        version = registry.register(
            name="model",
            base_model="base",
            adapter_path="/path",
            metrics={"initial": 1.0},
        )

        result = registry.update_metrics(
            version.version_id,
            {"new_metric": 0.9}
        )

        assert result is True

        updated = registry.get_version(version.version_id)
        assert updated.metrics["initial"] == 1.0
        assert updated.metrics["new_metric"] == 0.9

    # Best Model Tests
    def test_get_best_version(self, registry):
        """Test getting best version by metric."""
        registry.register(
            name="low",
            base_model="base",
            adapter_path="/p1",
            metrics={"avg_score": 0.5},
        )
        best = registry.register(
            name="high",
            base_model="base",
            adapter_path="/p2",
            metrics={"avg_score": 0.9},
        )
        registry.register(
            name="mid",
            base_model="base",
            adapter_path="/p3",
            metrics={"avg_score": 0.7},
        )

        best_version = registry.get_best_version(metric="avg_score")

        assert best_version.version_id == best.version_id

    # Persistence Tests
    def test_persistence(self, temp_dir):
        """Test that registry persists across instances."""
        # Create and populate registry
        registry1 = ModelRegistry(registry_dir=temp_dir)
        version = registry1.register(
            name="persistent",
            base_model="base",
            adapter_path="/path",
            set_active=True,
        )

        # Create new instance and verify data persists
        registry2 = ModelRegistry(registry_dir=temp_dir)

        assert registry2.get_version(version.version_id) is not None
        assert registry2.get_active() is not None

    # Comparison Tests
    def test_compare_versions(self, registry):
        """Test comparing multiple versions."""
        v1 = registry.register(
            name="v1",
            base_model="base",
            adapter_path="/p1",
            metrics={"score": 0.8, "loss": 0.2},
        )
        v2 = registry.register(
            name="v2",
            base_model="base",
            adapter_path="/p2",
            metrics={"score": 0.9, "loss": 0.1},
        )

        comparison = registry.compare_versions([v1.version_id, v2.version_id])

        assert v1.version_id in comparison
        assert v2.version_id in comparison
        assert comparison[v1.version_id]["metrics"]["score"] == 0.8
        assert comparison[v2.version_id]["metrics"]["score"] == 0.9

