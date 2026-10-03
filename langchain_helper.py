from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from dotenv import load_dotenv
load_dotenv()

try:
    import streamlit as st
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or str(st.secrets.get("GOOGLE_API_KEY", ""))
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
vectordb_file_path = "faiss_index"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_db():
    loader = CSVLoader(
        file_path="codebasics_faqs.csv",
        source_column="prompt",
    )
    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=get_embeddings(),
    )
    vectordb.save_local(vectordb_file_path)


def answer_question(question: str):
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it in Streamlit Cloud → Settings → Secrets."
        )

    if not os.path.exists(vectordb_file_path):
        raise RuntimeError(
            "Knowledge base is not created yet. Click 'Create Knowledgebase' first."
        )

    vectordb = FAISS.load_local(
        vectordb_file_path,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    documents = vectordb.similarity_search(question, k=4)

    context = "\n\n".join(
        f"FAQ {i}:\n{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )

    prompt = f"""Given the following context and a question, generate an answer based on this context only.
If the answer is not found in the context, say "I don't know." Do not make up an answer.

CONTEXT:
{context}

QUESTION:
{question}
"""

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    response = llm.invoke(prompt)
    content = response.content
    return content if isinstance(content, str) else str(content)


def get_qa_chain():
    return _SimpleQA()


class _SimpleQA:
    def invoke(self, inputs):
        question = inputs["query"]
        return {"result": answer_question(question)}


if __name__ == "__main__":
    create_vector_db()
    print(answer_question("What is RAG?"))
