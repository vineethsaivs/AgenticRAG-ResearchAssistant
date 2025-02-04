# tools.py

from retriever import retrieve_documents
from ollama_client import chat
from config import OLLAMA_MODEL

def get_search_results(query: str) -> str:
    """
    Returns research paper content relevant to the given query.
    """
    results = retrieve_documents(query)
    return f"Retrieved research content:\n{results}"

def summarize_paper(text: str) -> str:
    """
    Summarizes the provided research paper text into a concise paragraph.
    """
    prompt = f"Summarize the following research paper text in a concise paragraph:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    response = chat(model=OLLAMA_MODEL, messages=messages)
    return response.get("message", {}).get("content", "")

def compare_papers(text1: str, text2: str) -> str:
    """
    Compares two research paper excerpts, highlighting key similarities and differences.
    """
    prompt = (
        f"Compare the following two research paper excerpts. "
        f"Highlight the main similarities and differences.\n\n"
        f"Text 1:\n{text1}\n\nText 2:\n{text2}"
    )
    messages = [{"role": "user", "content": prompt}]
    response = chat(model=OLLAMA_MODEL, messages=messages)
    return response.get("message", {}).get("content", "")

def analyze_citations(text: str) -> str:
    """
    Analyzes the citations in the provided research paper text,
    identifying key references and their significance.
    """
    prompt = (
        f"Analyze the citations in the following research paper excerpt. "
        f"Identify key references and explain their significance.\n\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    response = chat(model=OLLAMA_MODEL, messages=messages)
    return response.get("message", {}).get("content", "")
