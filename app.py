import os
import openai
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
#
_ = load_dotenv(find_dotenv())
#
openai.api_key  = os.getenv("OPENAI_API_KEY")
#
loader = PyPDFLoader("data/John_Saul-2015-Geologist-Speculates.pdf")
pages = loader.load()
#
# Let's review and test the relevant pages.
print(len(pages))

# Let's picka chunking strategy.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
    separators=["\n\n", "\n", " ", ""]
)
#
doc_chunks = text_splitter.split_documents(pages)
#
# Let's pick an embedding strategy.
embedding_model = OpenAIEmbeddings()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_API_ENV")
)
#
print("Pinecone initialized")
#
index_name = "saul-geologist-speculates"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric="cosine",
        dimension=1536
    )
    #
    print(f"Created index: {index_name}")
#
""" search_index = Pinecone.from_documents(
    documents=doc_chunks,
    embedding=embedding_model,
    index_name=index_name,
) """
#
docsearch = Pinecone.from_existing_index(index_name, embedding_model)
#
query = "who is the author of the book?"
docs = docsearch.similarity_search(query)
#
print(f"Resulting docs: {len(docs)}")
#