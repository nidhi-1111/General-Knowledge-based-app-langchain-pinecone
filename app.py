## Langchain imports
import langchain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

## Pinecone imports
import pinecone
from langchain_pinecone import PineconeVectorStore

## OpenAI imports 
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain import OpenAI

## Gemini pro imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import faiss
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Read the documents 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:  # Iterate over each PDF in the list
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pdf_data = pdf_doc.read()
            temp_file.write(pdf_data)
            pdf_reader = PdfReader(temp_file.name)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Divide the docs into chunks
def get_text_chunks(text,chunk_size=10000,chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, embeddings='google'):
    index_name = "langchain-vector"
    if embeddings=='google':
        embeddings =GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        vector_store = PineconeVectorStore.from_texts(text_chunks, embeddings, index_name=index_name)
    elif embeddings=='openai':
        embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
        vector_store = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5, request_timeout=120)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain = load_qa_chain(model, chain_type='stuff',prompt=prompt)
    return chain

def user_input(user_question,text_chunks):
    db = get_vector_store(text_chunks)
    docs = db.similarity_search(user_question, k=30)
    # Filter documents based on a minimum similarity score
    # filtered_docs = [doc for doc in docs if doc['score'] > 0.4]  # Adjust threshold as needed

    if docs:
        chain = get_conversational_chain()
        response = chain(
            {'input_documents': docs, 'question': user_question},
            return_only_outputs=True
        )
        print(response)
        st.write("Reply: ", response['output_text'])
    else:
        st.warning("No relevant documents found for your question.")

def main():
    st.set_page_config("Chat With Multiple PDFs")
    st.header("Chat with multiple PDFs using Gemini")

    uploaded_pdfs = st.sidebar.file_uploader("Upload your PDF files", accept_multiple_files=True)  # Allow multiple files

    if uploaded_pdfs is not None:  # Check if any PDFs are uploaded
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_pdfs)  # Pass the list of PDFs
            text_chunks = get_text_chunks(raw_text)

            user_question = st.text_input("Ask a Quesiton from the uploaded PDFs")
            if user_question:
                user_input(user_question, text_chunks)
            else:
                st.warning("Please enter a question about the uploaded PDFs.")
    else:
        st.warning("Please upload PDF files.")

if __name__ == '__main__':
    main()










# vectors = embeddings.embed_query("How are you?")
# len(vectors) will give you dimensions which you can enter in pinecone while creating index.

# Vector search DB in Pinecone
# index = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

# # Cosine similarity retreive the results from VectorDB
# def retrieve_query(query, k=2):
#     matching_results = index.similarity_search(query,k=k)
#     return matching_results

# """OPENAI model"""
# llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.5)

# # To make it Q-A chain
# chain = load_qa_chain(llm, chain_type='stuff')

# # Search answers from VectorDB
# def retrieve_answers(query):
#     doc_search = retrieve_query(query)
#     print(doc_search)
#     response = chain.run(input_documents=doc_search,question=query)
#     return response

# our_query = "what does president of the stripe has to say?"
# answer = retrieve_answers(our_query)
# print(answer)


# Read the documents 
# def read_doc(directory):
#     file_loader = PyPDFDirectoryLoader(directory)
#     documents = file_loader.load()
#     return documents
# doc = read_doc('documents/')
