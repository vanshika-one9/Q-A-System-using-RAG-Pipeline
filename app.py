import streamlit as st
import os
from docx import Document
from PyPDF2 import PdfReader

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# 🔐 Load API key securely
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# INPUT HANDLING 
def process_input(input_type, input_data):
    """Convert user input into plain text"""

    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        docs = loader.load()
        text = docs[0].page_content

    elif input_type == "PDF":
        pdf = PdfReader(input_data)
        text = "".join([page.extract_text() or "" for page in pdf.pages])

    elif input_type == "DOCX":
        doc = Document(input_data)
        text = "\n".join([p.text for p in doc.paragraphs])

    elif input_type == "TXT":
        text = input_data.read().decode("utf-8")

    elif input_type == "Text":
        text = input_data

    else:
        raise ValueError("Unsupported input")

    return text


# VECTOR STORE 
def create_vectorstore(text):
    """Split text → convert to embeddings → store in FAISS"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


#  QA SYSTEM 
def get_answer(vectorstore, query):
    """Ask question using retrieved context"""

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        temperature=0.5,
        max_length=512
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa.run(query)


# UI 
st.title("📚 RAG Q&A System")

input_type = st.selectbox("Choose input type", ["Link", "PDF", "DOCX", "TXT", "Text"])

# Take input
if input_type == "Link":
    data = st.text_input("Paste URL")

elif input_type in ["PDF", "DOCX", "TXT"]:
    data = st.file_uploader("Upload file")

else:
    data = st.text_area("Enter text")

# Process data
if st.button("Process"):
    if data:
        with st.spinner("Reading your data..."):
            text = process_input(input_type, data)
            st.session_state.vectorstore = create_vectorstore(text)
        st.success("Done! You can now ask questions.")
    else:
        st.warning("Please provide input")

# Ask question
query = st.text_input("Ask something from your data")

if query and "vectorstore" in st.session_state:
    with st.spinner("Thinking..."):
        answer = get_answer(st.session_state.vectorstore, query)
    st.write("### 🤖 Answer")
    st.write(answer)