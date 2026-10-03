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


def answer_question(
    question: str,
    k: int = 4,
    candidate_profile: str = "",
):
    validate_settings()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vectorstore = load_vectorstore()

    retrieval_query = question
    if candidate_profile.strip():
        retrieval_query = (
            f"{question}\nCandidate profile: {candidate_profile.strip()}"
        )

    docs = vectorstore.similarity_search(retrieval_query, k=k)

    if not docs:
        return {
            "answer": "I don't have that information in the current knowledge base.",
            "sources": [],
        }

    context = "\n\n".join(
        f"FAQ {i}:\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )

    profile_section = (
        candidate_profile.strip()
        if candidate_profile.strip()
        else "No candidate profile was provided."
    )

    prompt = f"""You are SAI-RAG, an educational course and technology advisor.

Use ONLY the supplied knowledge-base context for factual claims about courses,
course content, project features, or information stored by this application.

The user may ask about course buying, what to learn next, future technology,
prerequisites, or how a technology fits their candidate profile.

For course-buying questions, explain profile fit, prerequisites, what information
is missing, what should be verified before paying, and possible lower-cost paths
when supported by the context. Do not invent course-specific facts.

For future-technology questions, personalize the discussion to the candidate
profile and explain trade-offs, prerequisites, skill overlap, project value, and
a practical learning path. Do not declare one technology universally best.

If a requested course-specific or technology-specific fact is not in the
knowledge base, say:
"I don't have that information in the current knowledge base."

Never invent prices, discounts, certificates, placement guarantees, salary
figures, dates, links, or policies.

Prefer these sections for recommendation-style questions:
1. Profile fit
2. What to learn/check
3. Why it fits or what gap it fills
4. Before buying
5. Suggested next step

CANDIDATE PROFILE:
{profile_section}

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

        response = _llm("gemini-3.5-flash-lite").invoke(prompt)

    content = response.content
    answer = content if isinstance(content, str) else str(content)

    return {
        "answer": answer.strip(),
        "sources": [doc.page_content for doc in docs],
    }
