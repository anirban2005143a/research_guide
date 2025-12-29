
import logging
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace , HuggingFaceEndpointEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import MultiQueryRetriever , EnsembleRetriever , ContextualCompressionRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

"""# function for multi query retriver"""

def get_multiquery_documents(
    query: str,
    persist_directory: str = "./chroma_db",
    collection_name: str = "collection_research_guide",
    k: int = 5
) -> List[Document]:
    """
    Retrieves unique documents from ChromaDB using a Multi-Query approach.
    """
    # 1. Initialize Embeddings
    embedding_model = HuggingFaceEndpointEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2',
        huggingfacehub_api_token=os.environ['HF_TOKEN']
    )

    # 2. Load Vector Store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    # 3. Initialize HF Model for Query Generation & Compression
    llm = HuggingFaceEndpoint(
        repo_id="allenai/Olmo-3-7B-Instruct",
        huggingfacehub_api_token=os.environ['HF_TOKEN'],
        temperature=1.2
    )
    model = ChatHuggingFace(llm=llm)

    # 5. Multi-Query Prompting
    prompt = f"""
      ### ROLE ###
      You are an expert Information Retrieval Specialist. Your goal is to optimize user questions for Vector Database searches.

      ### TASK ###
      Generate EXACTLY 5 high-quality search queries based on the User Question.
      You must expand abbreviations (Full Forms), include common acronyms (Short Forms), correct spelling, and extract core technical key points.

      ### OUTPUT RULES ###
      - OUTPUT ONLY THE QUERIES.
      - NO numbering (1, 2, 3...).
      - NO bullet points.
      - NO introductory text (e.g., "Here are your queries...").
      - NO concluding text.
      - Each query MUST be on a new line.
      - Each query must vary in granularity (some broad, some highly specific).

      ### EXAMPLE INPUT/OUTPUT ###
      User Question: "How do LLMs use RAG for AI?"
      Output:
      Large Language Models Retrieval Augmented Generation Artificial Intelligence
      LLM RAG AI architecture and implementation
      document retrieval processes in generative ai models
      context injection for transformer based language models
      RAG vs fine-tuning for internal data knowledge

      ### USER QUESTION TO PROCESS ###
      {query}
      """

    generated_queries = model.invoke(prompt).content

    # print(generated_queries)

    # retrievers
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    docs = []

    for q in generated_queries.split('\n'):
        search_results = retriever.invoke(q)
        docs.extend(search_results)

    return docs



"""# function for hybrid search"""

def get_hybrid_documents(query, documents_list, persist_directory="./chroma_db", collection_name = "collection_research_guide", k=25):
    """
    Performs a hybrid search combining BM25 (Keyword) and Chroma (Vector).
    - query: The user string
    - documents_list: A list of LangChain Document objects (needed for the BM25 index)
    """

    #1. Initialize Embeddings and LLM
    embedding_model = HuggingFaceEndpointEmbeddings(
        model = 'sentence-transformers/all-MiniLM-L6-v2',
        huggingfacehub_api_token=os.environ['HF_TOKEN']
    )

    # Load the existing Chroma vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    #2. Setup BM25 Retriever (Keyword)
    # This creates a local index of your documents for keyword matching
    bm25_retriever = BM25Retriever.from_documents(documents_list)
    bm25_retriever.k = k

    # 3. Create Ensemble Retriever (Hybrid)
    # weights=[0.5, 0.5] means equal importance to keyword and semantic
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Retrieve unique, ranked documents
    hybrid_docs = ensemble_retriever.invoke(query)

    return hybrid_docs