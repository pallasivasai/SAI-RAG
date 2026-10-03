# SAI-RAG 🧠

A Streamlit RAG application for educational Q&A, course-buying questions, and personalized technology-learning guidance.

## What SAI-RAG does

SAI-RAG retrieves relevant entries from its local FAQ knowledge base using FAISS and sentence-transformer embeddings, then sends the retrieved context to Gemini for a grounded answer.

It supports questions such as:
- What is RAG or FAISS?
- What should I check before buying a course?
- Is a course relevant to my current skills?
- What technology should I learn next?
- How can a technology choice fit my target role?
- What prerequisites should I complete before choosing a course?
- What information is missing before making a course decision?

## Candidate-profile suggestions

The sidebar includes an optional Candidate Profile box. Enter your current level, skills, experience, target role, and interests.

Example profile:
> MCA fresher, Python/SQL/JavaScript, interested in cybersecurity and data, looking for entry-level roles.

Then ask:
> Which technology should I learn next and is this course worth considering for my profile?

The assistant uses the profile together with retrieved knowledge-base context to explain fit, gaps, prerequisites, trade-offs, and what to verify before paying. It does not claim that one technology is universally best and it does not guarantee employment outcomes.

## Free-only design

This project is intentionally configured for the free Gemini API path. The Gemini model is fixed in config.py so an old GEMINI_MODEL Streamlit secret cannot accidentally switch the application to another model.

Free-tier usage still has request limits, so free does not mean unlimited usage.

## Project structure

- app.py — Streamlit user interface
- config.py — configuration and free-only Gemini model selection
- create_vector_db.py — CSV loading, embeddings, and FAISS creation
- rag_pipeline.py — retrieval, profile-aware prompting, and Gemini generation
- data/sai_faqs.csv — knowledge-base content
- .streamlit/config.toml — Streamlit configuration
- requirements.txt — Python dependencies
- runtime.txt — Python runtime

## Streamlit deployment

Set the Streamlit entrypoint to app.py.

In Streamlit Cloud → Settings → Secrets, the required secret is:

GEMINI_API_KEY = your_gemini_api_key

The app intentionally fixes its Gemini model in code for the free-only project requirement.

After deployment, click Create Knowledge Base once to build the FAISS index for that runtime.

## Local usage

pip install -r requirements.txt
streamlit run app.py

## Updating the knowledge base

1. Edit data/sai_faqs.csv.
2. Commit and deploy the changes.
3. Click Create Knowledge Base in the app.
4. Ask a question again.

## Answer quality

The assistant is instructed not to invent course prices, discounts, certificates, placement guarantees, salary figures, dates, links, or policies. If a course-specific fact is missing from the knowledge base, it says so instead of presenting an unsupported claim.

## Attribution

This repository is an independent implementation inspired by the general RAG workflow demonstrated in the Codebasics LangChain project. It does not redistribute the original project's code, FAQ dataset, notebook, or image verbatim.
