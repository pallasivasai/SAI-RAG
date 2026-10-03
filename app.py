import streamlit as st

from create_vector_db import build_vector_store
from rag_pipeline import answer_question
from config import VECTORSTORE_DIR

st.set_page_config(page_title="SAI-RAG", page_icon="🧠", layout="centered")

st.title("🧠 SAI-RAG")
st.caption("Retrieval-Augmented Educational Q&A Assistant")

with st.sidebar:
    st.subheader("Knowledge Base")
    if st.button("Create Knowledge Base", use_container_width=True):
        with st.spinner("Building FAISS knowledge base..."):
            try:
                build_vector_store()
                st.success("Knowledge base created successfully.")
            except Exception as exc:
                st.error(str(exc))

    if VECTORSTORE_DIR.exists():
        st.success("FAISS index is ready.")
    else:
        st.warning("Knowledge base not created yet.")

question = st.text_input(
    "Ask a question",
    placeholder="Example: What skills are covered in the course?",
)

if question:
    with st.spinner("Searching the knowledge base..."):
        try:
            result = answer_question(question)
            st.subheader("Answer")
            st.write(result["answer"])

            if result["sources"]:
                with st.expander("Retrieved source context"):
                    for index, source in enumerate(result["sources"], start=1):
                        st.markdown(f"**Source {index}**")
                        st.write(source)
        except Exception as exc:
            st.error(str(exc))
