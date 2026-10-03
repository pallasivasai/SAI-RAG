import streamlit as st

from create_vector_db import build_vector_store
from rag_pipeline import answer_question, clear_rag_cache
from config import VECTORSTORE_DIR

st.set_page_config(
    page_title="SAI-RAG",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 SAI-RAG")
st.caption("RAG-powered Course, Career & Technology Advisor")

with st.sidebar:
    st.subheader("Knowledge Base")

    if st.button("Create Knowledge Base", use_container_width=True):
        with st.spinner("Building FAISS knowledge base..."):
            try:
                build_vector_store()
                clear_rag_cache()
                st.success("Knowledge base created successfully.")
            except Exception as exc:
                st.error(str(exc))

    if VECTORSTORE_DIR.exists():
        st.success("FAISS index is ready.")
    else:
        st.warning("Knowledge base not created yet.")

    st.divider()
    st.subheader("Candidate Profile")
    st.caption("Optional: add your profile so SAI-RAG can personalize course and technology suggestions.")

    candidate_profile = st.text_area(
        "Profile",
        placeholder=(
            "Example: MCA fresher, Python/SQL/JavaScript, interested in cybersecurity "
            "and data, looking for entry-level roles."
        ),
        height=130,
        label_visibility="collapsed",
    )

st.subheader("Ask SAI-RAG")

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Question",
        placeholder=(
            "Example: Should I buy this cybersecurity course, or should I learn "
            "another technology for my career?"
        ),
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask SAI-RAG", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the knowledge base and preparing your answer..."):
            try:
                result = answer_question(
                    question,
                    candidate_profile=candidate_profile,
                )

                st.subheader("Answer")
                st.write(result["answer"])

                if result["sources"]:
                    with st.expander("Retrieved source context"):
                        for index, source in enumerate(result["sources"], start=1):
                            st.markdown(f"**Source {index}**")
                            st.write(source)
            except Exception as exc:
                st.error(str(exc))
