import streamlit as st
from langchain_helper import create_vector_db, answer_question

st.set_page_config(page_title="SAI-RAG", page_icon="🧠")

st.title("SAI-RAG Q&A 🧠")
st.caption("Retrieval-Augmented Generation Q&A")

if st.button("Create Knowledgebase"):
    with st.spinner("Creating knowledge base..."):
        try:
            create_vector_db()
            st.success("Knowledgebase created successfully!")
        except Exception as exc:
            st.error(str(exc))

question = st.text_input("Question:", placeholder="Example: What is RAG?")

if question:
    with st.spinner("Searching and generating answer..."):
        try:
            answer = answer_question(question)
            st.header("Answer")
            st.write(answer)
        except Exception as exc:
            st.error(str(exc))
