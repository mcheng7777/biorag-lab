# BioRAG-Lab – Product Requirements Document (PRD)

## Product Overview

BioRAG-Lab is a reinforcement-learning powered RAG web application for bioinformatics researchers.  

It helps users:  

1. Discover relevant papers on a topic.  
2. Identify suitable public datasets.  
3. Generate runnable R/Python code using Gemini API.  

## Target Users

- Computational biologists
- Bioinformatics researchers
- Students/educators in genomics

## Goals

- **Usefulness**: Provide real, runnable code for users to test.  
- **Accessibility**: No private data needed; relies on public datasets.  
- **RL Adaptation**: Improve retrieval + generation quality from user feedback.  
- **Deployment**: Web interface hosted on Hugging Face Spaces.  

## Core Features

1. **User Query Input**
   - Free-text natural language queries.

2. **Paper Retrieval**
   - Retrieve papers from PMC/arXiv.
   - Summarize with Gemini.

3. **Dataset Retrieval**
   - Retrieve datasets from GEO/SRA.
   - Provide metadata + links.

4. **Code Generation**
   - Use Gemini for R/Python code generation.
   - Validate code (syntax, imports, basic run).

5. **Embeddings**
   - Use Gemini embeddings for FAISS index.
   - Retrieve relevant context.

6. **RL Reranking**
   - Optimize search + code quality using user feedback.

7. **User Feedback**
   - Store in Supabase DB.
   - Used in PPO training loop.

8. **Web Interface**
   - Built with Next.js + Shadcn UI.
   - Authentication with Supabase.
   - Show papers, datasets, and generated code with copy/download buttons.

9. **CI/CD**
   - GitHub Actions for linting, testing, deploying to Hugging Face Spaces.
