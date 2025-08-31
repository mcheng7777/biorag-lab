# BioRAG Lab

A reinforcement-learning powered RAG application for bioinformatics researchers. BioRAG Lab helps researchers discover relevant papers, identify suitable public datasets, and generate runnable R/Python code using the Gemini API.

## 🚀 Features

- **Paper Discovery**: Search and explore papers from PMC and arXiv
- **Dataset Integration**: Find relevant datasets from GEO and SRA
- **Code Generation**: Generate R/Python code for bioinformatics analysis
- **RL-Powered**: Continuous improvement through user feedback
- **Modern Stack**: Next.js 14, FastAPI, Supabase, and Gemini API

## 📁 Project Structure

```
biorag-lab/
├── frontend/               # Next.js frontend application
│   ├── app/               # App router pages
│   ├── components/        # React components
│   └── lib/              # Utilities and helpers
│
├── backend/               # FastAPI backend service
│   ├── app/              # Application package
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Core functionality
│   │   ├── models/       # Data models
│   │   └── services/     # Business logic
│   └── tests/            # Test suite
│
└── docs/                 # Documentation
    ├── PLANNING.md       # Development roadmap
    ├── PRD.md           # Product requirements
    └── TECH_STACK.md    # Technical architecture
```

## 🛠 Tech Stack

### Frontend
- Next.js 14 with App Router
- TypeScript
- Shadcn UI + Tailwind CSS
- React Query

### Backend
- FastAPI
- FAISS for vector search
- LangChain + Gemini API
- uv for dependency management

### Infrastructure
- Supabase (Auth + Database)
- Docker
- GitHub Actions
- Hugging Face Spaces

## 🚦 Getting Started

1. **Prerequisites**
   - Docker and Docker Compose
   - Git
   
   For local development:
   - Python 3.13+
   - Node.js 20+

2. **Quick Start with Docker**
   ```bash
   # Clone the repository
   git clone https://github.com/mcheng7777/biorag-lab.git
   cd biorag-lab

   # Start development environment
   docker-compose up
   ```

3. **Local Development**
   ```bash
   # Frontend
   cd frontend
   npm install
   npm run dev

   # Backend
   cd backend
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

For detailed setup instructions, see [INSTALLATION.md](INSTALLATION.md).

## 📖 Documentation

- [Product Requirements Document](docs/PRD.md)
- [Technical Stack](docs/TECH_STACK.md)
- [Development Planning](docs/PLANNING.md)
- [Installation Guide](INSTALLATION.md)

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm run test
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)
- [Shadcn UI](https://ui.shadcn.com/)
- [Supabase](https://supabase.com/)
- [Gemini API](https://deepmind.google/technologies/gemini/)
