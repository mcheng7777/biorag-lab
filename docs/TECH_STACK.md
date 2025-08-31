# BioRAG-Lab – Technical Stack & Implementation

## Frontend

- **Next.js 14**
- **Shadcn UI (Tailwind)**
- **React Query / SWR** for API
- **Supabase Auth**
- **Code Editor**:
  - Monaco Editor for code display
  - Syntax highlighting for R/Python
  - Real-time validation
- **Deployment**: Hugging Face Spaces

## Backend

### Core Services
- **FastAPI**
- **FAISS** for retrieval
- **Uvicorn** for serving

### LLM & RAG
- **GPT-OSS-20B**:
  - Primary code generation model
  - Fine-tunable for specialized tasks
  - Apache 2.0 licensed
  - Runs on consumer hardware (16GB GPU)
- **Gemini API**:
  - Embeddings (`text-embedding-004`)
  - Summarization (`gemini-pro`)
  - Fallback code generation
- **LangChain** integration layer for RAG
- **LangGraph** for RAG pipeline

### Documentation Processing
- **Beautiful Soup / Scrapy**: Documentation scraping
- **MkDocs Parser**: Parse Python package docs
- **Rdoc Parser**: Parse R package docs
- **Pandoc**: Convert documentation formats

### Code Generation & Execution
- **Docker SDK**: Sandbox environment management
- **Jupyter Kernel Gateway**: Code execution
- **Language Servers**:
  - R Language Server
  - Python Language Server
- **Code Analysis**:
  - AST parsers for R/Python
  - Static analyzers
  - Type checkers

## Database

- **Supabase (Postgres + Auth)**
- Tables:
  - `users`
  - `queries`
  - `feedback`
  - `package_docs` (NEW)
  - `code_executions` (NEW)
  - `rl_training_data` (NEW)

## RL Component

### Training Infrastructure
- **Algorithm**: PPO
- **Framework**: Ray/RLlib
- **Distributed Training**: Hugging Face Accelerate / HPC

### Code Generation Models
- **Documentation-Based Model**:
  - Input: Package docs + user query
  - Output: Package-specific code
  - Reward: Code execution success + doc adherence

- **Paper Implementation Model**:
  - Input: Paper content + dataset + implementation type
  - Output: Research code implementation
  - Reward: Output validation + user feedback

### Reward Signals
- **Code Quality**:
  - Syntax correctness
  - Runtime performance
  - Memory usage
  - Code style adherence
- **Execution Success**:
  - Output validation
  - Error handling
  - Resource usage
- **User Feedback**:
  - Thumbs up/down
  - Code reuse metrics
  - Implementation accuracy

## Code Execution Environment

### Sandbox Components
- **Docker Containers**:
  - R environment (tidyverse, BioConductor)
  - Python environment (scipy, sklearn)
  - Resource limits
  - Network isolation
- **Execution Pipeline**:
  - Code validation
  - Dependency resolution
  - Output capture
  - Error handling

### Security Measures
- **Resource Limits**:
  - CPU/Memory caps
  - Execution timeouts
  - Storage limits
- **Network Controls**:
  - Allowlist for package repos
  - No external network access
- **Code Scanning**:
  - Static analysis
  - Vulnerability checks
  - Malware detection

## CI/CD

- **GitHub Actions**
  - Frontend: ESLint, Jest
  - Backend: Black, flake8, pytest
  - Container security scans
  - Build Docker images
  - Deploy to Hugging Face Spaces

## Monitoring & Analytics

- **Performance**:
  - Code execution metrics
  - Model inference times
  - Resource utilization
- **Quality**:
  - Code generation success rates
  - User satisfaction metrics
  - RL model convergence
- **Security**:
  - Sandbox breach attempts
  - Resource abuse detection
  - Code scan results

## Deployment

- Hugging Face Spaces (Docker)
- Frontend + backend served together
- Supabase hosted externally
- Gemini API provides LLM functionality
- Distributed training on HPC/cloud
- Sandbox environments in isolated VPC