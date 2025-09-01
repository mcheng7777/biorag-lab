# BioRAG-Lab – Planning Document

## Phase 1 – Project Setup ✓

### Frontend Setup ✓
- [x] Initialize Next.js project with TypeScript
- [x] Set up Tailwind CSS and Shadcn UI
- [x] Create base layout and navigation
- [x] Implement responsive design
- [x] Set up initial pages structure

### Frontend Components ✓
- [x] Create shared components (PageHeader, Cards, etc.)
- [x] Implement home page layout
- [x] Create papers page with search and filters
- [x] Create datasets page with search and filters
- [x] Create code playground with split view
- [x] Add loading states and animations
- [x] Implement error boundaries
- [x] Add toast notifications for actions

### Backend Setup ✓
- [x] Set up FastAPI with project structure
- [x] Configure uv for dependency management
- [x] Create initial API endpoints
- [x] Set up development environment

### AI Agent Development _(Next Priority)_
- [ ] Set up MCP server integration
  - [ ] Download and configure paper-search-mcp server
  - [ ] Set up MCP client for tool communication
  - [ ] Test MCP server connectivity
- [ ] Implement Gemini API integration
  - [ ] Set up Google Generative AI client
  - [ ] Configure Gemini Pro for agent reasoning
  - [ ] Implement conversation management
- [ ] Build LangGraph agent orchestration
  - [ ] Design agent workflow nodes
  - [ ] Implement tool calling with MCP server
  - [ ] Create conversation state management
  - [ ] Add error handling and retry logic
- [ ] Create agent API endpoints
  - [ ] Chat endpoint for user interactions
  - [ ] Tool execution endpoints
  - [ ] Conversation history management
  - [ ] Streaming responses for real-time interaction
- [ ] Add agent capabilities
  - [ ] Paper search and analysis
  - [ ] Code generation with context
  - [ ] Dataset exploration
  - [ ] Research summarization

### Backend API Development _(After Agent Setup)_
- [ ] Implement core API endpoints
  - [ ] Agent chat endpoint (`/agent/chat`)
  - [ ] Tool execution endpoint (`/agent/tools`)
  - [ ] Conversation management (`/agent/conversations`)
  - [ ] Enhanced health check with agent status
- [ ] Add proper error handling and logging
  - [ ] Global exception handlers
  - [ ] Structured logging with structlog
  - [ ] Request/response logging
- [ ] Implement API documentation
  - [ ] Enhanced OpenAPI documentation
  - [ ] Example requests and responses
  - [ ] API versioning strategy
- [ ] Add input validation and models
  - [ ] Pydantic models for requests/responses
  - [ ] Input validation and sanitization
  - [ ] Response serialization
- [ ] Test backend functionality locally
  - [ ] Unit tests for endpoints
  - [ ] Integration tests
  - [ ] Agent workflow tests

### Docker Setup _(After Backend Development)_
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