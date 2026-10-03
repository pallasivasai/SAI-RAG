from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()

try:
    import streamlit as st
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or str(st.secrets.get("GOOGLE_API_KEY", ""))
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
vectordb_file_path = "faiss_index"

instructor_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def create_vector_db():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it in Streamlit Cloud → Settings → Secrets.")

    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it in Streamlit Cloud → Settings → Secrets.")

    vectordb = FAISS.load_local(
        vectordb_file_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke({"query": "What is RAG?"}))
