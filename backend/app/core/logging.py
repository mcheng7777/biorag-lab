import sys
import structlog
from opentelemetry import trace
from prometheus_client import Counter, Histogram

# Initialize tracer
tracer = trace.get_tracer(__name__)

# Initialize metrics
MODEL_LOAD_TIME = Histogram(
    "model_load_seconds",
    "Time spent loading the model",
    buckets=[1, 5, 10, 30, 60, 120]
)

INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Time spent on model inference",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

INFERENCE_REQUESTS = Counter(
    "model_inference_requests_total",
    "Total number of inference requests",
    ["mode", "status"]
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)

# Create logger instance
logger = structlog.get_logger()
