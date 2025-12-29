from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Initialize Embeddings
embedding_model = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2',
    huggingfacehub_api_token=os.environ['HF_TOKEN']
)

# 2. Load Vector Store
vectorstore = Chroma(
    persist_directory="./chroma_db1",
    embedding_function=embedding_model,
    collection_name="collection_research_guide"
)

def process_file(file_path) :

    if not file_path :
      return

    # load the document
    loader = PyPDFLoader(file_path)
    docs_generator = loader.lazy_load()
    docs = []
    for doc in docs_generator:
      docs.append(doc)

    # split the documents in chunks
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        " ",
        "."
      ],
    )

    splitted_docs = splitter.split_documents(docs)

    # do embedding and store in the vector store
    vectorstore.add_documents(splitted_docs)

# print("processing ")
# process_file(r"pdfs\NIPS-2017-attention-is-all-you-need-Paper.pdf")

print("searching")
docs = vectorstore.similarity_search(query = "Underground water" , k=10)

for doc in docs :
  print(doc.page_content)

