# SAI-RAG

An end-to-end Retrieval-Augmented Generation (RAG) educational Q&A assistant inspired by the architecture demonstrated in the Codebasics Q&A project, rebuilt as an independent implementation using modern Gemini/LangChain-compatible tooling.

## Architecture

CSV knowledge base → document loading → embeddings → FAISS vector store → semantic retrieval → Gemini → grounded answer

## Features

- CSV-based FAQ knowledge base
- Local FAISS vector database
- Hugging Face sentence-transformer embeddings
- Google Gemini generation
- Context-grounded answers with an explicit fallback when information is unavailable
- Streamlit interface
- Rebuild knowledge base from the UI
- Source snippets shown with answers

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your Gemini API key.
4. Build the vector database:

```bash
python create_vector_db.py
```

5. Start the app:

```bash
streamlit run app.py
```

## Environment

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Never commit a real API key. The repository intentionally does not include `.env`.

## Project Structure

- `app.py` — Streamlit application
- `rag_pipeline.py` — retrieval + Gemini generation
- `create_vector_db.py` — creates FAISS index from CSV
- `config.py` — environment and path configuration
- `data/sai_faqs.csv` — sample FAQ knowledge base
- `requirements.txt` — dependencies
- `.env.example` — environment template
- `.gitignore` — excludes secrets and generated artifacts

## Attribution

This project is an independent recreation of the general RAG Q&A workflow demonstrated in the Codebasics LangChain project. The original tutorial/project used Google PaLM and older LangChain APIs; SAI-RAG uses a modernized implementation instead of copying those legacy dependencies.
