# Langchain imports
import asyncio
import bs4

from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import WebBaseLoader 
# from langchain.document_loaders.web import WebLoader
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Embedding and model import
# Other
import streamlit as st
import os
import time
from PyPDF2 import PdfReader
import tempfile
import pdfplumber
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')


st.title("Ask questions from your PDF(s) or website")
option = None

# Prompt user to choose between PDFs or website
option = st.radio("Choose input type:", ("PDF(s)", "Website"), index=None)

def get_pdf_processed(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                text += page.extract_text()
    return text



model_name = "all-MiniLM-L6-v2"
st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap= 200)

 
if option:
    if option == "Website":
        website_link = st.text_input("Enter the website link:")
        if website_link:
            with st.spinner("Loading website content..."):
                st.session_state.loader = WebBaseLoader(web_paths=(website_link,),
                        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                        class_=("post-title","post-content","post-header")
                        )))

                st.session_state.docs = st.session_state.loader.load()
                print(st.session_state.docs)
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
                st.session_state.vector = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)            
            st.success("Done!")
            
    elif option == "PDF(s)":
        pdf_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        if pdf_files:
            with st.spinner("Loading pdf..."):
                st.session_state.docs = get_pdf_processed(pdf_files)
                st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.docs)
                st.session_state.vector = FAISS.from_texts(st.session_state.final_documents,st.session_state.embeddings)            
            st.success("Done!")


def llm_model():

    llm = ChatGroq(model="mixtral-8x7b-32768",groq_api_key=groq_api_key)
    # llm = ChatGroq(model="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
    """
    )
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vector.as_retriever() if st.session_state.vector else None
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    prompt = st.text_input("Input your question here")

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input":prompt})
        st.write(response['answer'])
        st.write("Response time: ", time.process_time() - start)


if option:
    llm_model()
