import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS

import time
from PyPDF2 import PdfReader
import tempfile
from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
# groq_api_key = os.getenv('GROQ_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')

st.title("Ask questions from your pdf(s) or website")
option = None

# Prompt user to choose between PDFs or website
option = st.radio("Choose input type:", ("PDF(s)", "Website"), index=None)


def get_pdf_processed(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def llm_model():
    llm = ChatGroq(model="mixtral-8x7b-32768",groq_api_key=st.secrets['GROQ_API_KEY'])
    # llm = ChatGroq(model="mixtral-8x7b-32768",groq_api_key=groq_api_key)
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

        start =time.process_time()
        response = retrieval_chain.invoke({"input":prompt})
        print("Response time :", time.process_time()-start)
        st.write(response['answer'])

st.session_state.embeddings =GoogleGenerativeAIEmbeddings(model = 'models/embedding-001',google_api_key=st.secrets['GOOGLE_API_KEY'])
# st.session_state.embeddings =GoogleGenerativeAIEmbeddings(model = 'models/embedding-001',google_api_key=google_api_key)
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap= 200)

if option:
    if option == "Website":
        website_link = st.text_input("Enter the website link:")
        if website_link:
            with st.spinner("Loading..."):
                st.session_state.loader = WebBaseLoader(website_link)
                st.session_state.docs = st.session_state.loader.load()
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
                st.session_state.vector = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
                st.success("Done!")
                llm_model()
            
    elif option == "PDF(s)":
        pdf_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        if pdf_files:
            with st.spinner("Loading..."):
                st.session_state.docs = get_pdf_processed(pdf_files)
                st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.docs)
                st.session_state.vector = FAISS.from_texts(st.session_state.final_documents,st.session_state.embeddings)
                st.success("Done!")
                llm_model()

            

        # with st.expander("Not the expected answer? Find the different one here"):
        #     for i, doc in enumerate(response['context']):
        #         st.write(doc.page_content)
        #         st.write("-----------------------------")

        

