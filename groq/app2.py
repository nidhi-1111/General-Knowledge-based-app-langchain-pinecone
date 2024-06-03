import streamlit as st
import os
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec


import time
from PyPDF2 import PdfReader
import tempfile
from dotenv import load_dotenv
load_dotenv()
import psutil
import os
import pdfplumber

## Load the API keys
groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# google_api_key = os.getenv('GOOGLE_API_KEY')
# neo4j_uri = os.getenv('NEO4J_URI')
# neo4j_username = os.getenv('NEO4J_USERNAME')
# neo4j_pass = os.getenv('NEO4J_PASSWORD')
# graph = Neo4jGraph()


# option = None

# Prompt user to choose between PDFs or website
# option = st.radio("Choose input type:", ("PDF(s)", "Website"), index=None)
# Create an instance of the Pinecone class
pinecone = Pinecone(api_key=pinecone_api_key)
index_name = "myindex"


st.title("Ask questions from your PDF(s) or website")
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size =10000, chunk_overlap= 1000)


if "vector" not in st.session_state:
    st.session_state.vector = None

clear_index = st.checkbox("Clear existing index")

if clear_index:
    # Delete the existing index
    pinecone.delete_index(index_name)
    st.success("Index cleared!")

def get_pdf_processed(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                text += page.extract_text()
    return text


def llm_model(input_text):
    # llm = ChatGroq(model="mixtral-8x7b-32768",groq_api_key=st.secrets['GROQ_API_KEY'])
    llm = ChatGroq(model="mixtral-8x7b-32768",groq_api_key=groq_api_key)
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context.
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
    start = time.process_time()
    response = retrieval_chain.invoke({"input":input_text})
    st.write(response['answer'])
    st.write("Response time: ", time.process_time() - start)

# st.session_state.embeddings =GoogleGenerativeAIEmbeddings(model = 'models/embedding-001',google_api_key=st.secrets['GOOGLE_API_KEY'])

# vector = PineconeVectorStore(index_name=index_name, embedding=embeddings)
# Check if the index exists, and create it if it doesn't
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Adjust the dimension based on your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Adjust the region based on your preference
        )
    )

pdf_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    with st.spinner("Loading pdf..."):
        docs = get_pdf_processed(pdf_files) 
        final_documents = text_splitter.split_text(docs)
        if index_name not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=index_name,
                dimension=384,  # Adjust the dimension based on your embedding model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Adjust the region based on your preference
                )
            )
        st.session_state.vector = LangchainPinecone.from_texts(final_documents, index_name=index_name, embedding = embeddings) 
        st.success("Done!")

user_question = st.text_input("Input your question here")
if user_question:
    llm_model(user_question)


clear_session = st.button("Clear Session")

if clear_session:
    # Reset the session state
    st.session_state.clear()
    st.success("Session cleared!")




# if option:
    # if option == "Website": 
    #     website_link = st.text_input("Enter the website link:")
    #     if website_link:
    #         with st.spinner("Loading website content..."):
    #             st.session_state.loader = WebBaseLoader(website_link)
    #             st.session_state.docs = st.session_state.loader.load()
    #             st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    #             # st.session_state.vector = PineconeVectorStore(index_name=index_name, embedding=st.session_state.embeddings)
    #             st.session_state.vector = PineconeVectorStore.from_documents(st.session_state.final_documents, index_name=index_name, embedding = st.session_state.embeddings)
    #             st.success("Done!")
    #             prompt = st.text_input("Input your question here")
    #         if prompt :
    #             llm_model(prompt)

    # elif option == "PDF(s)":



        # with st.expander("Not the expected answer? Find the different one here"):
        #     for i, doc in enumerate(response['context']):
        #         st.write(doc.page_content)
        #         st.write("-----------------------------")

        
# from langchain_community.vectorstores.faiss import FAISS
# from langchain.vectorstores import Pinecone
# from langchain_community.embeddings import OllamaEmbeddings
# from sentence_transformers import SentenceTransformer
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
