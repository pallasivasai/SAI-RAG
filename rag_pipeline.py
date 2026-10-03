import csv
import re
from difflib import SequenceMatcher

import streamlit as st
from langchain_community.vectorstores import FAISS

from config import (
    FAQ_FILE,
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


@st.cache_data(show_spinner=False)
def _faq_rows():
    if not FAQ_FILE.exists():
        return []

    with FAQ_FILE.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _direct_faq_answer(question: str):
    normalized_question = _normalize(question)
    if not normalized_question:
        return None

    best_row = None
    best_score = 0.0

    for row in _faq_rows():
        prompt = row.get("prompt", "").strip()
        response = row.get("response", "").strip()
        if not prompt or not response:
            continue

        normalized_prompt = _normalize(prompt)

        if normalized_question == normalized_prompt:
            return {"answer": response, "sources": [f"{prompt}: {response}"]}

        score = SequenceMatcher(None, normalized_question, normalized_prompt).ratio()

        question_words = set(normalized_question.split())
        prompt_words = set(normalized_prompt.split())
        if question_words and prompt_words:
            overlap = len(question_words & prompt_words) / len(question_words | prompt_words)
            score = max(score, overlap)

        if score > best_score:
            best_score = score
            best_row = row

    # Only use the deterministic FAQ response for a strong match.
    if best_row and best_score >= 0.78:
        prompt = best_row["prompt"].strip()
        response = best_row["response"].strip()
        return {"answer": response, "sources": [f"{prompt}: {response}"]}

    return None


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Knowledge base is not created yet. Click 'Create Knowledge Base' first."
        )

    # Streamlit Cloud can restart with an old runtime vectorstore. Rebuild it
    # automatically whenever the FAQ CSV is newer than the FAISS index.
    index_file = VECTORSTORE_DIR / "index.faiss"
    if FAQ_FILE.exists() and index_file.exists():
        if FAQ_FILE.stat().st_mtime > index_file.stat().st_mtime:
            from create_vector_db import build_vector_store
            build_vector_store()

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
    _faq_rows.clear()
    load_vectorstore.clear()
    _llm.clear()


def _needs_personalized_reasoning(question: str, candidate_profile: str) -> bool:
    if not candidate_profile.strip():
        return False

    triggers = (
        "should i", "should i buy", "worth", "which", "what should",
        "recommend", "suggest", "best for me", "fit me", "for my profile",
        "next technology", "learn next", "career", "future", "roadmap",
        "course", "technology", "tech", "role"
    )
    lowered = question.lower()
    return any(trigger in lowered for trigger in triggers)


def answer_question(
    question: str,
    k: int = 4,
    candidate_profile: str = "",
):
    validate_settings()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Exact/near-exact FAQ questions do not need Gemini. This makes common
    # course/project questions fast and avoids consuming free-tier requests.
    direct_answer = _direct_faq_answer(question)
    if direct_answer and not _needs_personalized_reasoning(question, candidate_profile):
        return direct_answer

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
prerequisites, career direction, or how a technology fits their candidate profile.

Treat the candidate profile as a real decision context. Extract the user's
current foundation, transferable skills, target role, experience level,
projects/certifications, interests, constraints, and obvious skill gaps.

For personalized questions, do not give a generic internet-style list. Connect
the answer to the profile, explain what is already strong, identify the missing
capability, and then present a small number of practical learning paths.
Explain what each path builds on, what prerequisite it needs, what project could
demonstrate it, and what evidence the candidate should look for before paying
for a course.

Be creative in presentation while staying factual: use concise headings,
decision trees, "keep / add / avoid for now" guidance, mini roadmaps, project
ideas, or a 30/60/90-day learning sequence when the context supports it.
Do not invent facts about a course, employer, market, salary, certification,
placement, or technology. Do not claim that one path is universally best.

For course-buying questions, explain profile fit, prerequisites, what information
is missing, what should be verified before paying, and possible lower-cost paths
when supported by the context. Do not invent course-specific facts.

For future-technology questions, personalize the discussion to the candidate
profile and explain trade-offs, prerequisites, skill overlap, project value, and
a practical learning path. Do not declare one technology universally best.

If the context contains a direct answer to the user's question, answer from
that context instead of saying that the information is missing.

If a requested course-specific or technology-specific fact is not in the
knowledge base, say:
"I don't have that information in the current knowledge base."

Never invent prices, discounts, certificates, placement guarantees, salary
figures, dates, links, or policies.

Prefer these sections for recommendation-style questions:
1. What I understand about your profile
2. Your current advantage
3. The skill gap to solve
4. Possible paths (with trade-offs)
5. Course-buying checks, if relevant
6. Practical next step

If the profile is broad or mixed, preserve that breadth and explain how the
skills can connect instead of forcing the candidate into a single identity.
If the user is a fresher, prioritize foundations, demonstrable projects and
role alignment. If the user has relevant experience, focus on the next
capability that compounds their existing work.

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
