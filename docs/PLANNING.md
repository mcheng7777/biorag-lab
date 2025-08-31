# BioRAG-Lab – Planning Document

## Phase 1 – Project Setup _(In Progress)_

### Frontend Setup ✓
- [x] Initialize Next.js project with TypeScript
- [x] Set up Tailwind CSS and Shadcn UI
- [x] Create base layout and navigation
- [x] Implement responsive design
- [x] Set up initial pages structure

### Frontend Components _(In Progress)_
- [x] Create shared components (PageHeader, Cards, etc.)
- [x] Implement home page layout
- [x] Create papers page with search and filters
- [x] Create datasets page with search and filters
- [x] Create code playground with split view
- [ ] Add loading states and animations
- [ ] Implement error boundaries
- [ ] Add toast notifications for actions

### Backend Setup ✓
- [x] Set up FastAPI with project structure
- [x] Configure uv for dependency management
- [x] Create initial API endpoints
- [x] Set up development environment

### Docker Setup _(Paused)_
- [x] Create development Dockerfile
- [x] Create production Dockerfile
- [x] Set up docker-compose
- [ ] Test and debug container setup
- [ ] Optimize container builds

### Documentation ✓
- [x] Create PRD
- [x] Document technical stack
- [x] Create installation guide
- [x] Document development setup

## Phase 2 – Database

### Supabase Setup
- [ ] Create Supabase project
- [ ] Configure authentication
- [ ] Set up database schema:
  - users table
  - queries table
  - feedback table
- [ ] Add row-level security policies
- [ ] Set up backup strategy

### Frontend Integration
- [ ] Add Supabase client
- [ ] Implement authentication UI
- [ ] Add protected routes
- [ ] Handle auth state management

## Phase 3 – Retrieval Layer

### Data Ingestion
- [ ] Set up paper ingestion pipeline
  - PMC API integration
  - arXiv API integration
  - Paper metadata extraction
- [ ] Set up dataset ingestion pipeline
  - GEO API integration
  - SRA API integration
  - Dataset metadata extraction
- [ ] Store embeddings using Gemini
- [ ] Index with FAISS
- [ ] Implement caching strategy

## Phase 4 – Code Generation

- [ ] FastAPI `/generate_code` endpoint
- [ ] Gemini API integration
- [ ] Code validation:
  - Syntax checking
  - Import validation
  - Security scanning
- [ ] Support for multiple languages (R/Python)

## Phase 5 – Summarization

- [ ] Gemini integration for summarization
- [ ] Paper summary templates
- [ ] Dataset summary templates
- [ ] Display in UI:
  - Summary cards
  - Expandable details
  - Key findings highlight

## Phase 6 – RL Reranking

- [ ] PPO implementation:
  - User feedback collection
  - Code validation metrics
  - Relevance scoring
- [ ] FastAPI reranker integration
- [ ] A/B testing framework
- [ ] Performance monitoring

## Phase 7 – Frontend Polish

### UI Components
- [ ] Enhanced query box with suggestions
- [ ] Interactive paper/dataset cards
- [ ] Code editor with syntax highlighting
- [ ] Feedback system UI

### User Experience
- [ ] Add loading states
- [ ] Implement infinite scrolling
- [ ] Add keyboard shortcuts
- [ ] Implement search filters

## Phase 8 – CI/CD

### GitHub Actions Setup
- [ ] Frontend workflow:
  - ESLint checks
  - TypeScript type checking
  - Unit tests
  - Build verification
- [ ] Backend workflow:
  - Black formatting
  - Flake8 linting
  - Pytest execution
  - Docker build
- [ ] Deployment workflow:
  - Environment configuration
  - Hugging Face Spaces deployment
  - Post-deployment checks

### Testing
- [ ] Unit test suite
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Performance testing

### Monitoring
- [ ] Error tracking setup
- [ ] Performance monitoring
- [ ] Usage analytics
- [ ] Cost monitoring

## Notes
- Tasks marked with ✓ are completed
- Tasks marked with _(In Progress)_ are currently being worked on
- Tasks marked with _(Paused)_ are temporarily on hold
- Subtasks provide more granular tracking of progress