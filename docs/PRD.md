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
   
   A. Documentation-Based Code Generation
   - Accept package documentation URLs (e.g., ComplexHeatmap, seaborn)
   - Parse and understand package documentation
   - Generate code examples based on user queries using GPT-OSS-20B
   - Validate against package API specifications
   - Provide interactive examples with common use cases
   - Fine-tune model on package-specific documentation
   
   B. Paper Implementation Code Generation
   - Generate code based on selected papers and datasets
   - Support multiple implementation types:
     - Figure/plot reproduction
     - Statistical analysis replication
     - Method implementation from scratch
   - Link code to specific paper sections/figures
   - Include explanatory comments for methodology
   - Fine-tune model on paper-code pairs
   
   C. Code Validation & RL Training
   - Execute generated code in sandboxed environment
   - Validate outputs against expected results
   - Use execution success as RL reward signal
   - Collect user feedback on code quality
   - Fine-tune GPT-OSS-20B through RL:
     - Documentation adherence rewards
     - Code execution success rewards
     - User feedback integration
     - Continuous model improvement
   
   D. Model Architecture
   - Base model: GPT-OSS-20B (Apache 2.0 licensed)
   - Specialized fine-tuning for each generation path
   - Configurable reasoning levels (low/medium/high)
   - Built-in tool use capabilities
   - Local deployment support

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
