# retriever.py

import os
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from config import RESEARCH_PAPERS_DIR

def load_documents():
    """
    Loads research papers from the specified directory.
    Supports .txt and .pdf files.
    """
    documents = []
    if not os.path.exists(RESEARCH_PAPERS_DIR):
        raise ValueError(f"Directory {RESEARCH_PAPERS_DIR} does not exist.")

    for filename in os.listdir(RESEARCH_PAPERS_DIR):
        filepath = os.path.join(RESEARCH_PAPERS_DIR, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
        elif filename.endswith(".pdf"):
            try:
                from langchain_community.document_loaders import PyPDFLoader
            except ImportError:
                raise ImportError("Please install PyPDF2 and langchain with PDF support.")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
        else:
            print(f"Unsupported file format: {filename}. Skipping.")
    
    if not documents:
        raise ValueError("No research papers loaded for retrieval.")
    return documents

def initialize_retriever():
    """
    Loads and indexes research papers from the designated directory.
    """
    documents = load_documents()
    
    # Use a larger chunk size for research papers.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(doc_chunks, embedding=embeddings, collection_name="research_assistant")
    retriever = vectorstore.as_retriever()
    return retriever

def retrieve_documents(query: str) -> str:
    """
    Retrieves and formats relevant research paper text for the given query.
    """
    try:
        retriever = initialize_retriever()
    except ValueError as e:
        return f"Retriever error: {e}"
    
    results = retriever.get_relevant_documents(query)
    
    # Join all retrieved chunks.
    result_text = "\n\n".join([doc.page_content for doc in results])
    return result_text
