# SAI-RAG

An end-to-end Retrieval-Augmented Generation (RAG) educational Q&A assistant inspired by the architecture demonstrated in the Codebasics Q&A project, rebuilt as an independent implementation using modern Gemini and LangChain tooling.

## 🚀 Streamlit deployment

Deploy this repository with Streamlit Community Cloud.

**Repository:** `pallasivasai/SAI-RAG`  
**Branch:** `main`  
**Entrypoint:** `app.py`

### Streamlit Cloud setup

1. Open Streamlit Community Cloud and connect your GitHub account.
2. Create an app from `pallasivasai/SAI-RAG`.
3. Select branch `main` and entrypoint `app.py`.
4. Open **Advanced settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

5. Deploy. The app builds the FAISS index from the included CSV when you click **Create Knowledge Base**.

Never commit your real Gemini API key to GitHub. Use Streamlit Cloud Secrets for deployment.

## Architecture

CSV knowledge base → document loading → embeddings → FAISS vector store → semantic retrieval → Gemini → grounded answer

## Features

- CSV-based FAQ knowledge base
- Local FAISS vector database
- Hugging Face sentence-transformer embeddings
- Google Gemini generation
- Context-grounded answers
- Streamlit interface
- Rebuild knowledge base from the UI
- Retrieved source context shown with answers

## Local setup

```bash
git clone https://github.com/pallasivasai/SAI-RAG.git
cd SAI-RAG
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your Gemini API key, then:

```bash
python create_vector_db.py
streamlit run app.py
```

## Project Structure

- `app.py` — Streamlit application
- `rag_pipeline.py` — retrieval + Gemini generation
- `create_vector_db.py` — creates FAISS index from CSV
- `config.py` — environment/secret and path configuration
- `data/sai_faqs.csv` — sample FAQ knowledge base
- `requirements.txt` — dependencies
- `.streamlit/config.toml` — Streamlit configuration
- `.env.example` — local environment template
- `.gitignore` — excludes secrets and generated artifacts

## Attribution

This project is an independent recreation of the general RAG Q&A workflow demonstrated in the Codebasics LangChain project. The original tutorial/project used Google PaLM and older LangChain APIs; SAI-RAG uses a modernized implementation rather than copying those legacy dependencies.
