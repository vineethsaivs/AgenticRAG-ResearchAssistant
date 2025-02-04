# Agentic RAG Research Assistant

This repository demonstrates an **Agentic Retrieval-Augmented Generation (RAG)** approach to building a research assistant. It integrates a language model (invoked via [Ollama](https://github.com/jmorganca/ollama)), a vector database (Chroma), and specialized tools for retrieving, summarizing, comparing, and analyzing research papers.

---

## Features

1. **Tool-Based Retrieval**  
   - Uses a vector store (Chroma) to search large sets of research papers (PDFs or text).
   - Dynamically calls custom tools (summarize, compare, analyze citations).

2. **Agentic Workflow**  
   - The language model decides *if and when* to invoke tools based on user queries.
   - Reduces hallucinations by fetching relevant context from external sources.

3. **Conversational & One-Shot Modes**  
   - **Conversational CLI:** Engage with the assistant in a multi-turn conversation in your terminal.
   - **One-Shot Method** (`agent.run(...)`): Suitable for integration into web apps or other UIs (like Streamlit).

4. **Streamlit UI**  
   - Basic web interface for entering a query and receiving a final answer.

---

## File Overview

- **`requirements.txt`**  
  Lists all Python dependencies (LangChain, Chroma, PyPDF2, etc.).

- **`config.py`**  
  - `OLLAMA_MODEL` for specifying which Ollama model to use (default: `"llama3.1:8b"`).
  - `RESEARCH_PAPERS_DIR` for the folder containing research papers (PDFs or text).

- **`ollama_client.py`**  
  - A `chat` function that calls the Ollama CLI via `subprocess.run`, parsing JSON/text output.

- **`retriever.py`**  
  - Loads and splits PDF/text research papers into chunks.
  - Builds a Chroma vector store to enable semantic similarity search.
  - Provides a `retrieve_documents` function to fetch relevant text from the corpus.

- **`tools.py`**  
  - Defines tool functions that the language model can call:
    - `get_search_results`
    - `summarize_paper`
    - `compare_papers`
    - `analyze_citations`

- **`agent.py`**  
  - Implements the agent logic:
    - Maintains a conversation (list of messages).
    - Decides when to invoke tools (based on the model’s JSON instructions).
    - Offers:
      - **`run(user_query)`** for one-shot queries.
      - **`converse()`** for a multi-turn, interactive conversation in the terminal.

- **`main.py`**  
  - Entry point for the **Conversational CLI**. Initializes `Agent` and calls `agent.converse()`.

- **`app.py`**  
  - A **Streamlit** UI that provides a text input and displays the final answer returned by `agent.run()`.

---

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up the Ollama CLI**

   - Install [Ollama](https://github.com/jmorganca/ollama) and ensure you have the desired model (e.g., `llama3.1:8b`) downloaded.
   - Verify it works by running:
     ```bash
     ollama run llama3.1:8b
     ```
   - Update `OLLAMA_MODEL` in `config.py` if you use a different model name.

3. **Place Research Papers**

   - Create a folder named `research_papers` in your project directory (or rename according to `config.py`).
   - Add `.pdf` or `.txt` files containing research papers you want to query.

4. **(Optional) Provide OPENAI_API_KEY**

   - If using OpenAI embeddings, you need an `OPENAI_API_KEY`.  
   - Set it in your environment or pass it in your code:
     ```bash
     export OPENAI_API_KEY="your-openai-api-key"
     ```
   - Or switch to other embeddings from `langchain_community` if you prefer a local model.

---

## Usage

### 1. Conversational CLI

Run the assistant in interactive mode:

```bash
python main.py
```

- Type your questions at the prompt.
- Type `exit` or `quit` to end.

### 2. One-Shot Query (as used by Streamlit)

If you just want a single answer, you can call:

```python
final_answer = agent.run(user_query)
```

This is integrated in **`app.py`**, which uses Streamlit.

### 3. Streamlit UI

For a simple web-based UI:

```bash
streamlit run app.py
```

- A browser tab opens.  
- Type a query and hit **Submit** to see a final answer.

---

## FAQ

**Q: I see Deprecation Warnings from LangChain.**  
A: Either update your imports (to `langchain_community.*`) or suppress warnings with:
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```
We've already done this in some files.

**Q: Why do I need `OPENAI_API_KEY`?**  
A: The default embeddings (OpenAIEmbeddings) call the OpenAI API to get vectors. If you prefer local embeddings, switch to another class from `langchain_community`.

**Q: My model is calling unknown tool names.**  
A: We mapped synonyms in `agent.py`. If the model uses new/unexpected tool names, add them to `self.tool_mapping`.  

**Q: How do I add more features?**  
A: Just define new functions in `tools.py` and update `agent.py`. The LLM can call them if you prompt it with the correct tool schemas.

---

## Contributing

Contributions and pull requests are welcome! For significant changes, please open an issue first to discuss the proposal. Feel free to fork this repo and customize it for your own research flows or embedding approaches.

---

Enjoy building and exploring your **Agentic RAG Research Assistant**!

