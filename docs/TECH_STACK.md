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
- **Chat Interface**:
  - Real-time agent conversations
  - Streaming responses
  - Tool execution visualization
- **Deployment**: Hugging Face Spaces

## Backend

### Core Services
- **FastAPI**
- **FAISS** for retrieval
- **Uvicorn** for serving

### AI Agent Architecture
- **MCP (Model Context Protocol)**:
  - paper-search-mcp server for academic tools
  - Tool communication and execution
  - Extensible tool ecosystem
- **Google Gemini API**:
  - Gemini Pro for agent reasoning
  - Tool calling and conversation
  - Code generation with context
- **LangGraph**:
  - Agent workflow orchestration
  - State management and conversation flow
  - Error handling and retry logic
  - Streaming response management

### LLM & RAG (Future)
- **GPT-OSS-20B**:
  - Primary code generation model
  - Fine-tunable for specialized tasks
  - Apache 2.0 licensed
  - Runs on consumer hardware (16GB GPU)
- **Gemini API**:
  - Embeddings (`text-embedding-004`)
  - Summarization (`gemini-pro`)
  - Agent reasoning and tool calling
- **LangChain** integration layer for RAG
- **LangGraph** for agent orchestration

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
  - `conversations` (NEW)
  - `messages` (NEW)
  - `tool_executions` (NEW)
  - `queries`
  - `feedback`
  - `package_docs` (Future)
  - `code_executions` (Future)
  - `rl_training_data` (Future)

## Agent Capabilities

### Paper Search & Analysis
- **Multi-source search**: PubMed, arXiv, bioRxiv, CrossRef, etc.
- **Paper download**: PDF retrieval and text extraction
- **Research summarization**: AI-powered paper analysis
- **Citation analysis**: Reference tracking and impact assessment

### Code Generation
- **Context-aware generation**: Based on papers and user requirements
- **Multi-language support**: R, Python, and other languages
- **Documentation-based**: Using package documentation
- **Paper implementation**: Converting research to code

### Dataset Exploration
- **Dataset discovery**: Finding relevant datasets
- **Metadata analysis**: Understanding dataset structure
- **Integration guidance**: How to use datasets with code

### Interactive Conversations
- **Multi-turn discussions**: Extended research conversations
- **Tool execution**: Real-time tool calling and results
- **Conversation memory**: Maintaining context across turns
- **Streaming responses**: Real-time agent responses

## RL Component (Future)

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
  - Documentation quality
- **User Satisfaction**:
  - Conversation quality
  - Tool execution success
  - Response relevance
- **Research Impact**:
  - Paper relevance
  - Citation accuracy
  - Implementation fidelity

## Deployment & Infrastructure

### Development
- **Local Development**: Docker Compose
- **Hot Reloading**: FastAPI + Next.js
- **Environment Management**: Python venv + Node.js

### Production
- **Backend**: Docker containers on cloud platform
- **Frontend**: Static export to CDN
- **Database**: Supabase managed PostgreSQL
- **Agent Services**: Containerized MCP server
- **Monitoring**: Application performance monitoring

### CI/CD
- **GitHub Actions**: Automated testing and deployment
- **Code Quality**: ESLint, TypeScript, Python linting
- **Testing**: Unit tests, integration tests, agent workflow tests
- **Deployment**: Automated deployment to staging/production

## Security & Privacy

### Authentication
- **Supabase Auth**: JWT-based authentication
- **Role-based access**: User permissions and roles
- **API Security**: Rate limiting and request validation

### Data Privacy
- **Conversation encryption**: End-to-end message encryption
- **Tool execution isolation**: Sandboxed tool execution
- **Data retention**: Configurable conversation history
- **GDPR compliance**: Data deletion and export capabilities

## Performance & Scalability

### Caching
- **Redis**: Session and conversation caching
- **CDN**: Static asset delivery
- **Database**: Query result caching

### Scaling
- **Horizontal scaling**: Multiple backend instances
- **Load balancing**: Traffic distribution
- **Database scaling**: Read replicas and connection pooling
- **Agent scaling**: Multiple agent instances with load balancing

## Monitoring & Observability

### Logging
- **Structured logging**: JSON-formatted logs
- **Log aggregation**: Centralized log management
- **Error tracking**: Automated error reporting

### Metrics
- **Application metrics**: Response times, error rates
- **Agent metrics**: Tool execution success rates
- **User metrics**: Conversation quality and satisfaction
- **Business metrics**: Usage patterns and feature adoption

### Alerting
- **Performance alerts**: Response time thresholds
- **Error alerts**: Error rate monitoring
- **Availability alerts**: Service health monitoring