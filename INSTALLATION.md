# BioRAG-Lab Installation Guide

## Prerequisites

- Docker and Docker Compose
- Git

For local development without Docker (optional):
- Python 3.13+ (we're using 3.13.3)
- Node.js (to be installed)

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

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate
```

#### 3. Install Backend Dependencies
```bash
# Install dependencies using uv
uv pip install -r requirements.txt
```

Current backend dependencies (as of initial setup):
- fastapi
- uvicorn[standard]
- pydantic
- pydantic-settings
- python-dotenv
- supabase
- pytest (dev)
- black (dev)
- flake8 (dev)
- httpx

Future dependencies (to be added in later phases):
- langchain (Phase 3)
- langchain-google-genai (Phase 4)
- faiss-cpu (Phase 3)
- tenacity (Phase 4)

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
# To be added:
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GEMINI_API_KEY=your_gemini_api_key
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
uvicorn app.main:app --reload
```

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

## Notes

- The project uses a monorepo structure with separate `frontend` and `backend` directories
- Python dependencies are managed with `uv` for faster installation and better dependency resolution
- Development tools (black, flake8, pytest) are included in the main requirements.txt for consistency
- Additional dependencies will be added incrementally as we implement each phase of the project
- Docker is the recommended way to run the project, but local setup is also supported