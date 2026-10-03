# SAI-RAG 🧠

> **An AI-powered Retrieval-Augmented Generation (RAG) application for learning, career, course, and project guidance.**

SAI-RAG is a practical **Generative AI / RAG project** built to show how a modern AI application can combine a custom knowledge base, semantic search, vector retrieval, prompt engineering, and an LLM.

The main purpose of this project is the **AI engineering workflow**. Course, career, technology, and project guidance are the use cases built on top of that workflow.

---

## 🚀 What is SAI-RAG?

At a high level, SAI-RAG follows this pipeline:

```text
User Question + Optional Candidate Profile
                    ↓
               RAG Pipeline
                    ↓
          Semantic Similarity Search
                    ↓
        Sentence-Transformer Embeddings
                    ↓
                   FAISS
                    ↓
          Relevant Context Retrieved
                    ↓
                LangChain
                    ↓
                Gemini LLM
                    ↓
             Grounded AI Response
                    ↓
     Learning / Course / Career / Project Guidance
```

### In simple words

**You ask a question → SAI-RAG searches its own knowledge base → retrieves relevant information → gives that context to Gemini → Gemini generates the response.**

This is the core idea of **Retrieval-Augmented Generation**.

---

## 🧠 Why did I build this?

SAI-RAG is not intended to be only a chatbot.

It is a practical project for learning and implementing the building blocks used in modern AI applications:

- Generative AI
- LLM integration
- Retrieval-Augmented Generation (RAG)
- vector embeddings
- semantic search
- FAISS vector retrieval
- LangChain
- prompt engineering
- knowledge-base grounding
- profile-aware AI responses
- Streamlit AI application development

The project is structured so each layer can be understood and explained during development or an interview.

---

## 🔍 What makes it RAG?

A basic LLM application can look like this:

```text
User Question → LLM → Answer
```

SAI-RAG adds a retrieval layer before generation:

```text
User Question
      ↓
Create / use semantic representation
      ↓
Search the vector database
      ↓
Retrieve relevant knowledge
      ↓
Build a grounded prompt
      ↓
Send context + question to the LLM
      ↓
Generate the final response
```

That means the application can base its response on information stored in its own knowledge base instead of depending only on the model's general knowledge.

---

## 🏗️ Architecture

### High-level view

```text
                         SAI-RAG
                            │
              ┌─────────────┴─────────────┐
              │                           │
          AI ENGINE                  USE CASES
              │                           │
       ┌──────┼──────┐             ┌──────┼──────┐
       │      │      │             │      │      │
      RAG    LLM   Retrieval     Course  Career  Project
       │      │      │
       │      │   Semantic Search
       │      │      │
       │    Gemini   FAISS
       │      │      │
       └── LangChain┘
              │
      Sentence-Transformer
           Embeddings
```

### End-to-end flow

1. The user enters a question.
2. The user can optionally provide a candidate profile.
3. The application prepares the retrieval query.
4. Sentence-transformer embeddings represent the text semantically.
5. FAISS searches for similar knowledge.
6. The most relevant context is retrieved.
7. LangChain connects the retrieval and LLM workflow.
8. Gemini receives the retrieved context and the user question.
9. Prompt instructions guide the response to stay grounded in the supplied context.
10. The final answer and retrieved source context are displayed in Streamlit.

---

## 🧩 Technology Stack

| Layer | Technology | What it does |
|---|---|---|
| UI | **Streamlit** | Runs the web interface |
| Programming | **Python** | Core application logic |
| LLM | **Gemini** | Generates natural-language responses |
| Orchestration | **LangChain** | Connects retrieval, prompting, and generation |
| Embeddings | **Sentence Transformers** | Converts text into semantic vectors |
| Vector Search | **FAISS** | Finds similar vectors efficiently |
| Knowledge Source | **CSV knowledge base** | Stores the information used for grounding |
| Deployment | **Streamlit Cloud** | Hosts the application |

These technologies are connected as one AI pipeline rather than being isolated libraries.

---

## 📁 Project Structure

```text
SAI-RAG/
│
├── app.py
│   └── Streamlit user interface
│
├── config.py
│   └── API and model configuration
│
├── create_vector_db.py
│   └── Loads knowledge-base data, creates embeddings,
│       and builds the FAISS vector store
│
├── rag_pipeline.py
│   └── Retrieval, profile-aware prompting, and Gemini generation
│
├── data/
│   └── sai_faqs.csv
│       └── Knowledge-base content
│
├── .streamlit/
│   └── config.toml
│       └── Streamlit configuration
│
├── requirements.txt
│   └── Python dependencies
│
├── runtime.txt
│   └── Python runtime
│
├── .env.example
│   └── Example environment configuration
│
└── README.md
    └── Project documentation
```

---

## 🔬 What does each important file do?

### `app.py`

The front end of SAI-RAG.

It provides the:
- SAI-RAG interface
- knowledge-base creation control
- optional candidate profile input
- question input
- AI answer display
- retrieved context display

Streamlit Cloud uses this file as the application entry point.

### `create_vector_db.py`

This file creates the vector database.

```text
CSV Knowledge Base
       ↓
Load Documents
       ↓
Create Embeddings
       ↓
Store Vectors
       ↓
FAISS Index
```

The generated FAISS index becomes the retrieval layer used by the RAG pipeline.

### `rag_pipeline.py`

This is the main AI engine.

It handles:
- loading the FAISS vector store
- semantic retrieval
- optional profile context
- prompt construction
- Gemini generation
- returning the answer and retrieved sources

This is where the main **RAG workflow** happens.

### `config.py`

This file centralizes configuration such as:
- Gemini API key
- Gemini model configuration
- embedding model
- project paths
- knowledge-base location

### `data/sai_faqs.csv`

This is the application's knowledge base.

The text is converted into embeddings so that semantically relevant information can be retrieved by FAISS.

---

## 👤 Candidate Profile

SAI-RAG supports an optional **Candidate Profile** so the same RAG pipeline can produce more contextual guidance.

Example:

> MCA fresher | Python, SQL, JavaScript | interested in cybersecurity and data | entry-level roles

The profile can provide context about:
- education
- current skills
- experience level
- interests
- target role
- projects and certifications
- learning constraints

Conceptually:

```text
Question
   +
Candidate Profile
   ↓
SAI-RAG Retrieval + Reasoning
   ↓
Context-aware AI Response
```

The profile is additional reasoning context; it does not replace retrieval from the knowledge base.

---

## 💡 What can I ask SAI-RAG?

### AI / RAG

> What is RAG?

> Why is FAISS used here?

> What are embeddings?

> How does semantic search work?

### Learning

> What should I learn before starting a technology?

> What prerequisites should I complete?

### Course

> What should I verify before buying a course?

> Does this learning path fit my current background?

### Career

> How can I connect a technology to a target role?

> What skill gap should I work on next?

### Project

> What can I build after learning this technology?

> How can I turn learning into portfolio evidence?

These are application use cases. The underlying RAG architecture remains the same.

---

## 🛡️ Grounded AI

SAI-RAG follows a simple principle:

> **Retrieve relevant information first, then ask the LLM to reason over that context.**

The prompt instructs the model not to invent unsupported details about:
- prices
- discounts
- certificates
- placement guarantees
- salaries
- dates
- links
- policies

When a requested course-specific fact is not available in the knowledge base, the application is instructed to say:

> "I don't have that information in the current knowledge base."

This does not guarantee zero hallucinations, but it demonstrates an important RAG engineering technique: **grounding generation in retrieved context**.

---

## 🎯 The learning journey behind SAI-RAG

The project can be understood as a progression:

```text
RAG Fundamentals
      ↓
Document Retrieval
      ↓
Embeddings
      ↓
FAISS Vector Search
      ↓
LangChain Integration
      ↓
Gemini LLM Integration
      ↓
Prompt Engineering
      ↓
Profile-aware Responses
      ↓
Course / Career / Project Use Cases
```

This makes the repository easy to explain: each feature is built on top of the previous AI layer.

---

## 🧪 Example: what happens internally?

Suppose the user asks:

> "Which technology should I learn next?"

And enters:

> "MCA fresher, Python and SQL, interested in cybersecurity."

SAI-RAG follows this process:

```text
1. Receive the question
          ↓
2. Add optional profile context
          ↓
3. Search the knowledge base
          ↓
4. Create / use semantic embeddings
          ↓
5. Find similar knowledge with FAISS
          ↓
6. Retrieve relevant context
          ↓
7. Build a grounded prompt
          ↓
8. Send prompt + context to Gemini
          ↓
9. Generate the response
          ↓
10. Show the answer + retrieved context
```

That is the core **RAG engineering pattern** implemented by the project.

---

## 🆓 Free-only design

SAI-RAG is intentionally configured around a free Gemini API path where practical.

The Gemini model selection is controlled in `config.py` so an old `GEMINI_MODEL` Streamlit secret cannot accidentally switch the application to another model.

> **Free does not mean unlimited.** API providers can still impose request, token, or quota limits.

---

## ☁️ Streamlit Cloud deployment

The application is designed to run on Streamlit Cloud.

### Entry point

```text
app.py
```

### Required secret

In **Streamlit Cloud → Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

After deployment:

1. Open the application.
2. Click **Create Knowledge Base**.
3. Wait for the FAISS index to be created.
4. Optionally enter a candidate profile.
5. Ask a question.
6. SAI-RAG retrieves relevant context and generates the response.

---

## 💻 Run locally

```bash
git clone https://github.com/pallasivasai/SAI-RAG.git
cd SAI-RAG
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` in your environment before running the application.

---

## 🔄 Updating the knowledge base

1. Edit `data/sai_faqs.csv`.
2. Commit and deploy the updated file.
3. Open the Streamlit application.
4. Click **Create Knowledge Base**.
5. Ask the question again.

The FAISS vector store is regenerated from the updated knowledge base.

---

## 🧭 Future AI-engineering direction

The current project already demonstrates the core RAG pipeline.

Natural next extensions could include:

- PDF / document upload and ingestion
- website knowledge ingestion
- source citations and metadata
- conversation memory
- hybrid retrieval
- retrieval reranking
- query classification
- RAG evaluation and quality metrics
- advanced document chunking
- AI-generated project roadmaps

These are future extensions, not required for the current implementation.

---

## 📌 How to explain SAI-RAG in an interview

A clear technical explanation is:

> **"I built an AI-powered Retrieval-Augmented Generation application that uses sentence-transformer embeddings for semantic representation, FAISS for vector retrieval, LangChain for orchestration, and Gemini for grounded response generation. I then added profile-aware reasoning so the same RAG pipeline can provide contextual learning, course, career, and project guidance."**

That explanation describes the **engineering first** and the use cases second.

---

## 📚 Attribution

This repository is an independent implementation inspired by the general RAG workflow demonstrated in the Codebasics LangChain project.

It does **not** redistribute the original project's code, FAQ dataset, notebook, or image verbatim.

---

## 👨‍💻 Author

**P Siva Sai**

GitHub: https://github.com/pallasivasai

Project: https://github.com/pallasivasai/SAI-RAG