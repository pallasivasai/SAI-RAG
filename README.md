# SAI-RAG

An independent Streamlit RAG Q&A project using the same high-level project layout and workflow style as the Codebasics Q&A tutorial.

## Project structure

- `main.py` — Streamlit application
- `langchain_helper.py` — LangChain, FAISS, embeddings, and Gemini logic
- `codebasics_faqs.csv` — FAQ knowledge base
- `google_palm_codebasics_q_and_a.ipynb` — project notebook
- `faiss_index/` — generated FAISS vector database
- `requirements.txt` — Python dependencies
- `.env` — local environment template; never put a real API key in GitHub

## Streamlit Cloud

Set the Streamlit entrypoint to `main.py`.

In Streamlit Cloud → Settings → Secrets, add:

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

Then deploy/reboot and click **Create Knowledgebase**.

## Local usage

```bash
pip install -r requirements.txt
streamlit run main.py
```

The application creates a local FAISS index from `codebasics_faqs.csv`, retrieves relevant FAQ entries, and generates a grounded answer with Gemini.

## Attribution

This repository is an independent implementation inspired by the general RAG workflow demonstrated in the Codebasics LangChain project. It does not redistribute the original project's code, FAQ dataset, notebook, or image verbatim.
