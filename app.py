import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma model document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

uploaded_file = st.file_uploader(
    "Upload your file",
    type="pdf"
)
def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(
            model='models/gemini-embedding-001',
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        st.session_state.loader = PyPDFLoader(temp_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



if uploaded_file:
    vector_embeddings()
    st.write("Your PDF is uploaded")

prompt1 = st.text_input("Chat with your PDF")
import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    #with st.expander("Document Similarity Search"):
        # Find the relevant chunks
    #   for i, doc in enumerate(response["context"]):
    #        st.write(doc.page_content)
    #        st.write("--------------------------------")

    