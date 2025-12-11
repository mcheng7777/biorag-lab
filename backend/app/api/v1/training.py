"""
API endpoints for training and model management.

Provides:
- Training job management
- Dataset preparation
- Model selection and switching
- Training status monitoring
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/training", tags=["training"])


# Request/Response models
class DataCollectionRequest(BaseModel):
    """Request to collect training data."""

    sources: list[str] = Field(
        default=["github", "bioconductor", "biopython"],
        description="Data sources to collect from",
    )
    max_examples: int = Field(default=1000, ge=10, le=10000)
    github_token: Optional[str] = Field(default=None, description="GitHub API token")


class DataCollectionResponse(BaseModel):
    """Response from data collection."""

    status: str
    examples_collected: int
    output_path: str
    collection_time_seconds: float


class TrainingRequest(BaseModel):
    """Request to start a training job."""

    dataset_path: str = Field(..., description="Path to training dataset")
    base_model: str = Field(
        default="codellama/CodeLlama-7b-hf",
        description="Base model to fine-tune",
    )
    output_name: str = Field(..., description="Name for the fine-tuned model")
    # Training parameters
    num_epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=2048, ge=128, le=8192)


class TrainingResponse(BaseModel):
    """Response from training job."""

    job_id: str
    status: str
    started_at: str
    message: str


class TrainingStatusResponse(BaseModel):
    """Training job status."""

    job_id: str
    status: str
    progress: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models."""

    models: list[dict]
    active_model: Optional[str] = None


class ModelSelectRequest(BaseModel):
    """Request to select a model."""

    version_id: str = Field(..., description="Model version ID to activate")


class GenerateCodeRequest(BaseModel):
    """Request to generate code."""

    instruction: str = Field(..., description="What code to generate")
    context: str = Field(default="", description="Additional context")
    language: str = Field(default="python", description="Target language")
    max_tokens: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateCodeResponse(BaseModel):
    """Generated code response."""

    code: str
    language: str
    model_id: str
    generation_time_ms: float


# In-memory job tracking (in production, use a database)
_training_jobs: dict = {}


@router.post("/prepare-data", response_model=DataCollectionResponse)
async def prepare_training_data(
    request: DataCollectionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Collect and prepare training data from various sources.

    Collects bioinformatics code from:
    - GitHub repositories
    - Bioconductor package vignettes
    - Biopython documentation
    """
    import time

    from ...services.training import (
        DataCollector,
        DataPreprocessor,
        DataValidator,
    )

    start_time = time.time()

    # Collect data
    collector = DataCollector(
        github_token=request.github_token,
        output_dir="data/training",
    )

    examples = await collector.collect_all(
        max_examples=request.max_examples,
        sources=request.sources,
    )

    # Validate
    validator = DataValidator()
    valid_examples, _, stats = validator.validate_batch(examples)

    # Preprocess
    preprocessor = DataPreprocessor()
    training_examples = preprocessor.preprocess_batch(valid_examples)

    # Save
    from ...services.training import DatasetBuilder

    builder = DatasetBuilder(output_dir="data/datasets")
    train, val, test = preprocessor.split_dataset(training_examples)
    dataset = builder.build_dataset_dict(train, val, test)
    output_path = builder.save_dataset(dataset, name="bioinfo_code")

    elapsed = time.time() - start_time

    return DataCollectionResponse(
        status="completed",
        examples_collected=len(training_examples),
        output_path=str(output_path),
        collection_time_seconds=elapsed,
    )


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a fine-tuning training job.

    The training runs in the background. Use /training/status/{job_id}
    to monitor progress.
    """
    from ...services.training import TrainingConfig, LoRAConfig

    # Create training config
    lora_config = LoRAConfig(
        r=request.lora_r,
        lora_alpha=request.lora_alpha,
    )

    config = TrainingConfig(
        base_model=request.base_model,
        output_dir=f"models/fine_tuned/{request.output_name}",
        run_name=request.output_name,
        num_train_epochs=request.num_epochs,
        per_device_train_batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        max_seq_length=request.max_seq_length,
        lora=lora_config,
    )

    # Generate job ID
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Store job info
    _training_jobs[job_id] = {
        "status": "pending",
        "config": config,
        "dataset_path": request.dataset_path,
        "started_at": datetime.now().isoformat(),
        "metrics": None,
        "error": None,
    }

    # Start training in background
    async def run_training():
        from datasets import load_from_disk

        from ...services.training import BioinfoCodeTrainer

        try:
            _training_jobs[job_id]["status"] = "running"

            # Load dataset
            dataset = load_from_disk(request.dataset_path)

            # Train
            trainer = BioinfoCodeTrainer(config=config)
            result = trainer.train(
                train_dataset=dataset["train"],
                eval_dataset=dataset.get("validation"),
            )

            _training_jobs[job_id]["status"] = "completed"
            _training_jobs[job_id]["metrics"] = result.metrics

            # Register model
            from ...services.models import ModelRegistry

            registry = ModelRegistry()
            registry.register(
                name=request.output_name,
                base_model=request.base_model,
                adapter_path=result.output_dir,
                metrics=result.metrics,
                set_active=True,
            )

        except Exception as e:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_training)

    return TrainingResponse(
        job_id=job_id,
        status="pending",
        started_at=_training_jobs[job_id]["started_at"],
        message="Training job started. Use /training/status/{job_id} to monitor.",
    )


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = _training_jobs[job_id]

    return TrainingStatusResponse(
        job_id=job_id,
        status=job["status"],
        metrics=job.get("metrics"),
        error=job.get("error"),
    )


@router.get("/jobs")
async def list_training_jobs():
    """List all training jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "started_at": job["started_at"],
            }
            for job_id, job in _training_jobs.items()
        ]
    }


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available fine-tuned models."""
    from ...services.models import ModelRegistry

    registry = ModelRegistry()
    versions = registry.list_versions(limit=50)

    active = registry.get_active()

    return ModelListResponse(
        models=[v.to_dict() for v in versions],
        active_model=active.version_id if active else None,
    )


@router.post("/models/select")
async def select_model(request: ModelSelectRequest):
    """Select and activate a model version."""
    from ...services.models import get_inference_service

    service = get_inference_service()
    success = service.switch_model(request.version_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to switch to model {request.version_id}",
        )

    return {
        "status": "success",
        "active_model": request.version_id,
    }


@router.post("/generate", response_model=GenerateCodeResponse)
async def generate_code(request: GenerateCodeRequest):
    """
    Generate bioinformatics code using the active model.

    Requires a model to be loaded. Use /training/models to see
    available models and /training/models/select to activate one.
    """
    from ...services.models import GenerationRequest, get_inference_service

    service = get_inference_service()

    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Use /training/models/select to load a model.",
        )

    gen_request = GenerationRequest(
        instruction=request.instruction,
        context=request.context,
        language=request.language,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    response = await service.generate_async(gen_request)

    return GenerateCodeResponse(
        code=response.code,
        language=response.language,
        model_id=response.model_id,
        generation_time_ms=response.generation_time_ms,
    )


@router.get("/status")
async def get_service_status():
    """Get the status of the training and inference services."""
    from ...services.models import get_inference_service

    service = get_inference_service()

    return {
        "inference_service": service.get_status(),
        "training_jobs": {
            "total": len(_training_jobs),
            "running": sum(1 for j in _training_jobs.values() if j["status"] == "running"),
            "completed": sum(1 for j in _training_jobs.values() if j["status"] == "completed"),
            "failed": sum(1 for j in _training_jobs.values() if j["status"] == "failed"),
        },
    }

