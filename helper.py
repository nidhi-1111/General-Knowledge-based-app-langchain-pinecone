## Langchain imports
import langchain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

## Pinecone imports
import pinecone
from langchain_pinecone import PineconeVectorStore

## OpenAI imports 
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain import OpenAI

## Gemini pro imports

import os

from dotenv import load_dotenv
load_dotenv()

# Read the documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents
doc = read_doc('documents/')

# Divide the docs into chunks
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc
documents = chunk_data(docs=doc) 

"""If you are using openai model and embeddings"""
# Embedding Technique of OPENAI
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

vectors = embeddings.embed_query("How are you?")
# len(vectors) will give you dimensions which you can enter in pinecone while creating index.

# Vector search DB in Pinecone
index_name = "langchain-vector"
index = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

# Cosine similarity retreive the results from VectorDB
def retrieve_query(query, k=2):
    matching_results = index.similarity_search(query,k=k)
    return matching_results

"""OPENAI model"""
llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.5)

# To make it Q-A chain
chain = load_qa_chain(llm, chain_type='stuff')

# Search answers from VectorDB
def retrieve_answers(query):
    doc_search = retrieve_query(query)
    print(doc_search)
    response = chain.run(input_documents=doc_search,question=query)
    return response

our_query = "what does president of the stripe has to say?"
answer = retrieve_answers(our_query)
print(answer)
