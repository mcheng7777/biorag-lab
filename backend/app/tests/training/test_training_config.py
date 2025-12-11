"""Tests for training configuration classes."""

import pytest
from pydantic import ValidationError

from app.services.training.training_config import (
    LoRAConfig,
    ModelType,
    QuantizationType,
    TrainingConfig,
    get_qlora_config,
    get_full_finetune_config,
)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        assert len(config.target_modules) > 0

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
        )

        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1

    def test_invalid_rank(self):
        """Test validation of invalid rank."""
        with pytest.raises(ValidationError):
            LoRAConfig(r=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            LoRAConfig(r=300)  # Must be <= 256

    def test_invalid_dropout(self):
        """Test validation of invalid dropout."""
        with pytest.raises(ValidationError):
            LoRAConfig(lora_dropout=-0.1)  # Must be >= 0

        with pytest.raises(ValidationError):
            LoRAConfig(lora_dropout=0.6)  # Must be <= 0.5


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.base_model == ModelType.CODELLAMA_7B.value
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.bf16 is True
        assert config.gradient_checkpointing is True

    def test_custom_model(self):
        """Test configuration with custom model."""
        config = TrainingConfig(
            base_model=ModelType.STARCODER2_7B.value,
        )

        assert config.base_model == "bigcode/starcoder2-7b"

    def test_quantization_options(self):
        """Test quantization configuration."""
        config_nf4 = TrainingConfig(quantization=QuantizationType.NF4)
        config_int8 = TrainingConfig(quantization=QuantizationType.INT8)
        config_none = TrainingConfig(quantization=QuantizationType.NONE)

        assert config_nf4.quantization == QuantizationType.NF4
        assert config_int8.quantization == QuantizationType.INT8
        assert config_none.quantization == QuantizationType.NONE

    def test_invalid_epochs(self):
        """Test validation of invalid epochs."""
        with pytest.raises(ValidationError):
            TrainingConfig(num_train_epochs=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            TrainingConfig(num_train_epochs=150)  # Must be <= 100

    def test_invalid_batch_size(self):
        """Test validation of invalid batch size."""
        with pytest.raises(ValidationError):
            TrainingConfig(per_device_train_batch_size=0)  # Must be >= 1

    def test_invalid_learning_rate(self):
        """Test validation of invalid learning rate."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0)  # Must be > 0

        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-1e-4)  # Must be > 0

    def test_get_training_args_dict(self):
        """Test conversion to training arguments dict."""
        config = TrainingConfig(
            output_dir="test_output",
            num_train_epochs=5,
            learning_rate=1e-4,
        )

        args_dict = config.get_training_args_dict()

        assert args_dict["output_dir"] == "test_output"
        assert args_dict["num_train_epochs"] == 5
        assert args_dict["learning_rate"] == 1e-4
        assert "per_device_train_batch_size" in args_dict
        assert "gradient_accumulation_steps" in args_dict

    def test_nested_lora_config(self):
        """Test nested LoRA configuration."""
        lora = LoRAConfig(r=8, lora_alpha=16)
        config = TrainingConfig(lora=lora)

        assert config.lora.r == 8
        assert config.lora.lora_alpha == 16


class TestConfigPresets:
    """Tests for configuration presets."""

    def test_qlora_config(self):
        """Test QLoRA configuration preset."""
        training_config, lora_config = get_qlora_config()

        assert training_config.quantization == QuantizationType.NF4
        assert training_config.bf16 is True
        assert training_config.gradient_checkpointing is True
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32

    def test_full_finetune_config(self):
        """Test full fine-tuning configuration preset."""
        config = get_full_finetune_config()

        assert config.quantization == QuantizationType.NONE
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 16


class TestModelTypes:
    """Tests for model type enums."""

    def test_model_type_values(self):
        """Test model type enum values."""
        assert ModelType.CODELLAMA_7B.value == "codellama/CodeLlama-7b-hf"
        assert ModelType.CODELLAMA_13B.value == "codellama/CodeLlama-13b-hf"
        assert ModelType.STARCODER2_7B.value == "bigcode/starcoder2-7b"

    def test_quantization_type_values(self):
        """Test quantization type enum values."""
        assert QuantizationType.NONE.value == "none"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.NF4.value == "nf4"

