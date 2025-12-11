# BioRAG-Lab Installation Guide

## Prerequisites

- Docker and Docker Compose
- Git

For local development without Docker (optional):
- Python 3.12 (we're using 3.12.11)
- Node.js 20 LTS
- uv package manager

## Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd biorag-lab

# Start the development environment
docker-compose up --build
```

The API will be available at http://localhost:8000

## Docker Development

### Backend Development
```bash
# Build and start backend service
docker-compose up backend

# Run tests
docker-compose run --rm backend pytest

# Format code
docker-compose run --rm backend black .

# Lint code
docker-compose run --rm backend flake8
```

### Production Build
```bash
# Build production image
docker build -t biorag-lab-backend:prod -f backend/Dockerfile.prod ./backend

# Run production container
docker run -p 8000:8000 biorag-lab-backend:prod
```

## Local Development (Without Docker)

### Backend Setup

#### 1. Install uv (Python Package Manager)
```bash
# Install uv using the official installation script
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Set up Python Environment
```bash
# Navigate to backend directory
cd backend

# clone paper search mcp
git clone https://github.com/openags/paper-search-mcp.git
# Create virtual environment
uv init --python=3.12

```

#### 3. Install Backend Dependencies
```bash
# Install dependencies using uv
uv add paper-search-mcp
# This will create a .venv file. Activate the virtual environment and install other dependencies
source .venv/bin/activate
uv add -r requirements.txt
```

Current backend dependencies (as of GPT-OSS integration):

Core Dependencies:
- Python 3.12.11
- PyTorch 2.8.0
- Transformers 4.56.0
- FastAPI 0.116.1
- PydanticAI & pydantic-settings
- structlog (for structured logging)

Model & Training:
- Accelerate (for distributed training)
- DeepSpeed (for model optimization)
- Datasets (for data loading)
- Evaluate (for model evaluation)
- Wandb (for experiment tracking)
- Safetensors (for model weights)

Development Tools:
- Black (code formatting)
- Flake8 (linting)
- Pytest (testing)
- httpx (HTTP client for testing)

Model Serving:
- transformers-serve (for local development)
- vLLM (for production deployment - Linux only)

Future Dependencies:
- LangChain & LangGraph (for RAG pipeline)
- FAISS (for vector search)
- Gemini API (for embeddings and fallback)

## Frontend Setup

### 1. Install Node.js (via Homebrew)
```bash
# Install Node.js 20 LTS
brew install node@20

# Add Node.js to your PATH (if using zsh)
echo 'export PATH="/opt/homebrew/opt/node@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
node --version  # Should show v20.x.x
npm --version   # Should show v10.x.x
```

### 2. Set up Next.js Project
```bash
# Navigate to frontend directory
cd frontend

# Create Next.js project with TypeScript and Tailwind CSS
npx create-next-app@latest . \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --no-src-dir \
  --import-alias "@/*" \
  --yes
```

### 3. Install and Configure Shadcn UI
```bash
# Initialize Shadcn UI
npx shadcn@latest init

# When prompted:
# - Style: Default
# - Base color: Stone (or your preference)
# - CSS variables: Yes
# - Import alias: @/*
```

### 4. Current Frontend Dependencies
- Next.js 14
- React & React DOM
- TypeScript
- Tailwind CSS
- Shadcn UI
- ESLint

Additional dependencies to be added:
- React Query (for API integration)
- Supabase Client (for authentication)
- Other UI components as needed

## Database Setup (To Be Completed)

### 1. Supabase Setup
TBD - Will document Supabase project setup and configuration

## Environment Variables

### Backend (.env)
```env
# Model Configuration
MODEL_MODEL_CONFIG__MODEL_ID=openai/gpt-oss-20b
MODEL_MODEL_CONFIG__MAX_NEW_TOKENS=512
MODEL_MODEL_CONFIG__TEMPERATURE=0.7
MODEL_MODEL_CONFIG__TOP_P=0.95

# Resource Limits
MODEL_MAX_MEMORY_GB=16.0
MODEL_MAX_GPU_MEMORY_GB=16.0

# Monitoring
MODEL_ENABLE_METRICS=true
MODEL_ENABLE_TRACING=true
MODEL_METRICS_PORT=9090

# Database (to be added)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Frontend (.env.local)
```env
# To be added:
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Development

### Running Backend
```bash
# With Docker
docker-compose up backend

# Without Docker (from backend directory with virtual environment activated)
uvicorn app.main:app --reload --port 8001 --log-level debug

# Development server will be available at:
# - API: http://localhost:8001
# - API Documentation: http://localhost:8001/docs
# - Metrics: http://localhost:9090/metrics (if enabled)
```

#### Development Server Options
- `--reload`: Enable auto-reload on code changes
- `--port 8001`: Use port 8001 (8000 might be used by other services)
- `--log-level debug`: Detailed logging for development
- `--workers 1`: Single worker for development (default)
- `--host 0.0.0.0`: Listen on all interfaces (if needed)

### Running Frontend
```bash
# With Docker (to be added)
docker-compose up frontend

# Without Docker (from frontend directory)
npm run dev
```

The frontend will be available at http://localhost:3000

## Testing

### Backend Tests
```bash
# With Docker
docker-compose run --rm backend pytest

# Without Docker (from backend directory)
pytest
```

### Frontend Tests
TBD - Will document frontend testing setup

## Code Quality

### Backend
```bash
# With Docker
docker-compose run --rm backend black .
docker-compose run --rm backend flake8

# Without Docker
black .
flake8
```

### Frontend
```bash
# Format and lint code
npm run lint

# Type check
npm run build  # This will also check types
```

## Troubleshooting

### Common Issues

#### Python Environment
- **Wrong Python Version**: Make sure you're using Python 3.12.x
  ```bash
  # Check Python version
  python --version
  
  # If wrong version, make sure you've activated the correct environment
  source backend/bin/activate
  ```

- **Package Installation Fails**: Try updating uv
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

#### Model Service
- **Out of Memory**: Reduce model configuration in `.env`:
  ```env
  MODEL_MAX_MEMORY_GB=8.0  # Reduce memory limit
  MODEL_MODEL_CONFIG__MAX_NEW_TOKENS=256  # Reduce token limit
  ```

- **Port Already in Use**: Change port in uvicorn command:
  ```bash
  uvicorn app.main:app --reload --port 8002  # Try different port
  ```

#### Development Server
- **Auto-reload Not Working**: Check file watch limits
  ```bash
  # macOS: Check current limits
  launchctl limit maxfiles
  
  # Increase if needed (temporary)
  sudo launchctl limit maxfiles 65536 200000
  ```

### Getting Help
- Check the [GitHub Issues](https://github.com/yourusername/biorag-lab/issues)
- Review logs: `uvicorn` outputs to stdout/stderr
- Enable debug logging: `--log-level debug`
- Check metrics endpoint: `/metrics` if enabled

## Notes

- The project uses a monorepo structure with separate `frontend` and `backend` directories
- Python dependencies are managed with `uv` for faster installation and better dependency resolution
- Development tools (black, flake8, pytest) are included in the main requirements.txt for consistency
- Additional dependencies will be added incrementally as we implement each phase of the project
- Docker is the recommended way to run the project, but local setup is also supported
- Model serving uses `transformers-serve` for local development and `vLLM` for production (Linux only)