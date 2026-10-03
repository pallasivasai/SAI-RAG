from langchain_community.vectorstores import FAISS

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    validate_settings,
)


def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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


def answer_question(question: str, k: int = 4):
    validate_settings()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return {"answer": "I don't know.", "sources": []}

    context = "\n\n".join(
        f"FAQ {i}:\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are SAI-RAG, an educational knowledge-base assistant.
Answer using only the supplied context.
Do not invent facts, prices, policies, dates, links, or guarantees.
When the answer is not supported by the context, say exactly: I don't know.
Keep the answer clear and useful.

CONTEXT:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

    response = llm.invoke(
        prompt.format_messages(context=context, question=question)
    )

    content = response.content
    answer = content if isinstance(content, str) else str(content)

    return {
        "answer": answer.strip(),
        "sources": [doc.page_content for doc in docs],
    }
