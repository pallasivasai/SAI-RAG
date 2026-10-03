import streamlit as st
from langchain_community.vectorstores import FAISS

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    validate_settings,
)


@st.cache_resource(show_spinner=False)
def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Knowledge base is not created yet. Click 'Create Knowledge Base' first."
        )

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def _llm(model_name: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    validate_settings()

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )


def clear_rag_cache():
    _embeddings.clear()
    load_vectorstore.clear()
    _llm.clear()


def answer_question(question: str, k: int = 4):
    validate_settings()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return {
            "answer": "I don't have that information in the current knowledge base.",
            "sources": [],
        }

    context = "\n\n".join(
        f"FAQ {i}:\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )

    prompt = f"""You are SAI-RAG, an educational question-answering assistant.

Answer the user's question using ONLY the supplied knowledge-base context.
If the question is about a course, course topic, course availability, technology,
project feature, or learning topic, answer from the relevant FAQ information.
Do not invent courses, syllabus items, prices, dates, policies, links, or guarantees.

If the supplied context does not contain the answer, say:
"I don't have that information in the current knowledge base."

Keep the answer clear and concise.

KNOWLEDGE-BASE CONTEXT:
{context}

USER QUESTION:
{question}
"""

    try:
        response = _llm(GEMINI_MODEL).invoke(prompt)
    except Exception as exc:
        message = str(exc).lower()
        quota_error = "429" in message or "quota" in message or "rate limit" in message

        if not quota_error or GEMINI_MODEL == "gemini-3.5-flash-lite":
            raise

        # Free-first fallback: try the lower-cost Flash-Lite model if another
        # configured Gemini model has exhausted its quota.
        response = _llm("gemini-3.5-flash-lite").invoke(prompt)

    content = response.content
    answer = content if isinstance(content, str) else str(content)

    return {
        "answer": answer.strip(),
        "sources": [doc.page_content for doc in docs],
    }
