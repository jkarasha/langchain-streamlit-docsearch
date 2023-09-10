import os
import openai
import pinecone
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
#
#_ = load_dotenv(find_dotenv())
#openai.api_key  = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]
#
# Set the page attributes
st.set_page_config(page_title="A geologist speculates", page_icon="🌎", layout="wide")
st.title("A geologist speculates")
#
def load_pdf():
    loader = PyPDFLoader("data/John_Saul-2015-Geologist-Speculates.pdf")
    pages = loader.load()
    return pages
#
def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    #
    doc_chunks = text_splitter.split_documents(pages)
    return doc_chunks
#
def init_vector_store(index_name):
    pinecone.init(
        #api_key=os.getenv("PINECONE_API_KEY"),
        #environment=os.getenv("PINECONE_API_ENV")
        #
        api_key = st.secrets["PINECONE_API_KEY"],
        environment = st.secrets["PINECONE_API_ENV"]
    )
    #
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=1536
        )
#
def load_vector_store(doc_chunks, index_name):
    init_vector_store(index_name)
    #
    embedding_model = OpenAIEmbeddings()
    _ = Pinecone.from_documents(
        documents=doc_chunks,
        embedding=embedding_model,
        index_name=index_name,
    )
#
def generate_response(input_query, index_name):
    #
    embedding_model = OpenAIEmbeddings()
    init_vector_store(index_name)
    # We don't always need to load_vector_store
    if 1==0:
        pages = load_pdf()
        doc_chunks = split_pages(pages)
        load_vector_store(doc_chunks, index_name)
    #
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    docsearch = Pinecone.from_existing_index(index_name, embedding_model)
    #
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )
    #
    result = qa_chain({"query": input_query})
    print(result)

    #return st.success(result)
    return st.write(result["result"])

## App Widgets
query_text = st.text_input('Ask a question:', placeholder='Who is the author?')
index_name = "saul-geologist-speculates"
#
generate_response(query_text, index_name)
