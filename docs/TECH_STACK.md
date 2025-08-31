# BioRAG-Lab – Technical Stack & Implementation

## Frontend

- **Next.js 14**
- **Shadcn UI (Tailwind)**
- **React Query / SWR** for API
- **Supabase Auth**
- **Deployment**: Hugging Face Spaces

## Backend

- **FastAPI**
- **FAISS** for retrieval
- **Gemini API**:
  - Embeddings (`text-embedding-004`)
  - Summarization (`gemini-pro`)
  - Code generation (`gemini-1.5-pro` or latest)
- **LangChain** integration layer for RAG
- **LangGraph** for RAG pipeline
- **Uvicorn** for serving

## Database

- **Supabase (Postgres + Auth)**
- Tables:
  - `users`
  - `queries`
  - `feedback`

## RL Component

- **Algorithm**: PPO
- **Reward Signal**:
  - User thumbs up/down
  - Code execution validity
  - Dataset relevance
- **Training infra**: Hugging Face Accelerate / HPC

## CI/CD

- **GitHub Actions**
  - Frontend: ESLint, Jest
  - Backend: Black, flake8, pytest
  - Build Docker image
  - Deploy to Hugging Face Spaces

## Deployment

- Hugging Face Spaces (Docker)
- Frontend + backend served together
- Supabase hosted externally
- Gemini API provides all LLM functionality
