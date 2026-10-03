from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

from config import FAQ_FILE, VECTORSTORE_DIR, EMBEDDING_MODEL


def build_vector_store() -> Path:
    if not FAQ_FILE.exists():
        raise FileNotFoundError(f"FAQ file not found: {FAQ_FILE}")

    from langchain_huggingface import HuggingFaceEmbeddings

    loader = CSVLoader(
        file_path=str(FAQ_FILE),
        source_column="prompt",
    )
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return VECTORSTORE_DIR


if __name__ == "__main__":
    path = build_vector_store()
    print(f"Vector database created at: {path}")
